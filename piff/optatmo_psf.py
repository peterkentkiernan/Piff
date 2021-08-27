# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: psf
"""

from __future__ import print_function

import numpy as np
from scipy.optimize import least_squares
from iminuit import Minuit
import tabulate as tab  #for Minuit printout

import galsim

from .interp import Interp
from .outliers import Outliers
from .psf import PSF
from .wavefront import Wavefront
from .params import Params
from .util import write_kwargs, read_kwargs, calculate_moments, get_moment_names
import pdb

class OptAtmoPSF(PSF):
    """A PSF class that uses a combination of a Fraunhofer optical wavefront and an
    atmospheric turbulence kernel.

    The OptAtmoPSF uses a three-step fit to form the PSF.  First, a model convolving an optical
    wavefront and a spatially constant turbulence kernel is fit to a subset of stars. The
    optical wavefronts are taken from out-of-focus engineering images and/or an optical
    ray-tracing (Zemax) models, adjusted with a small number of free parameters. Next a test sample
    of stars is fit individually to the optical model found in step 1, convolved with a atmospheric kernel,
    floating parameters of the kernel.  Finally, the parameters of the atmospheric kernel are interpolated
    across the entire field of view with a Gaussian Process.
    """

    def __init__(self, model, interp, outliers=None,
                 do_ofit=True, do_sfit=True, do_afit=True,
                 wavefront_kwargs=None,
                 ofit_double_zernike_terms=[[4, 1], [5, 3], [6, 3], [7, 1], [8, 1], [9, 1],
                                            [10, 1], [11, 1], [14, 1], [15, 1]],
                 ofit_initvalues= {"opt_L0":25.,"opt_size":1.0},
                 ofit_bounds={"opt_L0":[5.,100.0]},
                 ofit_fix=None,
                 ofit_type='shape',
                 ofit_nstars=500,
                 ofit_optimizer='iminuit',
                 ofit_shape_kwargs={'moment_list':['e0','e1','e2'],'weights':None,'systerrors':None},
                 ofit_pixel_kwargs=None,
                 sfit_pixel_kwargs={'pixel_radius':2.0},
                 sfit_optimizer='iminuit',
                 moment_kernel='Gaussian',
                 fov_radius=4500.):
        """
        :param model:       A Model instance used for modeling the Optical component of the PSF.
        :param interp:      An Interp instance used to interpolate the Atmospheric component across the field of view.
        :param outliers:    Optionally, an Outliers instance used to remove outliers.
                            [default: None]
        :param do_ofit:     Perform the Optical Wavefront & Constant Atmospheric Kernel fit [default: True]
        :param do_sfit:     Perform the Individual Star Atmospheric Kernel fit [default: True]
        :param do_afit:     Perform the Gaussian Process Atmospheric Kernel interpolation [default: True]
        :param wavefront_kwargs: Options for Reference Wavefront interpolation [default: None]
        :param ofit_double_zernike_terms:   A list of the double Zernike coefficients (pupil,focal) to use as free parameters. [default:=[[4, 1], [5, 3], [6, 3], [7, 1], [8, 1], [9, 1],
                                            [10, 1], [11, 1], [14, 1], [15, 1]]]
        :param ofit_type:    Type of optical fit, shape or pixel [default: shape]
        :param ofit_nstars:  Number of stars to use in Optical fit [default: 500]
        :param ofit_optimizer:    Optimizer to use for Optical fit, iminuit or least_squares [default: 'iminuit']
        :param ofit_initvalues:   Dictionary of initial values for Optical fit parameters [default: {"opt_L0":25.,"opt_size":1.0}]
        :param ofit_bounds:       Dictionary of bounds for Optical fit parameters [default: {"opt_L0":[5.,100.0]}]
        :param ofit_fix:          List of fixed parameters for Optical fit [default:None]
        :param ofit_shape_kwargs: Dictionary with options for shape Optical fit [default: {'moment_list':['e0','e1','e2'],'weights':None,'systerrors':None}]
        :param ofit_pixel_kwargs: Dictionary with options for pixel Optical fit [default: None]
        :param sfit_pixel_kwargs: Dictionary with options for single star pixel Optical fit [default: {'pixel_radius':2.0}]
        :param sfit_optimizer:    Optimizer to use for Single Star fit, iminuit or least_squares [default: 'iminuit']
        :param moment_kernel:     String with Kernel type for Moments: Gaussian or VonKarman [default: 'Gaussian']
        :param fov_radius:   Field of View radius in arcsec [default: 4500.]

        """
        self.model = model
        self.interp = interp
        self.outliers = outliers

        self.do_ofit = do_ofit
        self.do_sfit = do_sfit
        self.do_afit = do_afit

        # save wavefront kwargs
        self.wavefront_kwargs = wavefront_kwargs

        # other kwargs
        self.ofit_double_zernike_terms = ofit_double_zernike_terms
        self.ofit_nstars = ofit_nstars
        self.ofit_optimizer = ofit_optimizer
        self.ofit_initvalues = ofit_initvalues
        self.ofit_bounds = ofit_bounds
        self.ofit_fix = ofit_fix
        self.ofit_shape_kwargs = ofit_shape_kwargs
        self.ofit_pixel_kwargs = ofit_pixel_kwargs
        self.sfit_pixel_kwargs = sfit_pixel_kwargs
        self.sfit_optimizer = sfit_optimizer
        self.moment_kernel = moment_kernel
        self.fov_radius = fov_radius

        # no additional properties are used, so set to empty list
        self.extra_interp_properties = []

        # save options needed when reading .piff file
        self.kwargs = {
            # model and interp are junk entries that will be overwritten.
            'model': 0,
            'interp': 0,
            'outliers': 0,
            'do_ofit': self.do_ofit,
            'do_sfit': self.do_sfit,
            'do_afit': self.do_afit,
            'fov_radius': self.fov_radius,
#            'ofit_double_zernike_terms': self.ofit_double_zernike_terms,  #TODO: need to serialize these...
#            'wavefront_kwargs': self.wavefront_kwargs
        }
        # done

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        import piff

        kwargs = {}
        kwargs.update(config_psf)
        kwargs.pop('type', None)

        for key in ['model', 'interp']:
            if key not in kwargs:  # pragma: no cover
                # This actually is covered, but for some reason, codecov thinks
                # it isn't.
                raise ValueError( "%s field is required in psf field for type=Simple" % key)

        # make a Model object to use for the Optical fit and the Individual Star fitting
        model = piff.Model.process(kwargs.pop('model'), logger=logger)
        kwargs['model'] = model

        # make an Interp object to use for the atmospheric interpolation
        interp = piff.Interp.process(kwargs.pop('interp'), logger=logger)
        kwargs['interp'] = interp

        if 'outliers' in kwargs:
            outliers = piff.Outliers.process(kwargs.pop('outliers'), logger=logger)
            kwargs['outliers'] = outliers

        return kwargs

    def fit(self, stars, wcs, pointing, logger=None):
        """Fit OptAtmo PSF model to star data using three fitting steps

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing

        if len(stars) == 0:
            raise RuntimeError("No stars.  Cannot find PSF model.")

        if self.do_ofit:
            logger.info("Step 1: Optical Wavefront & Constant Turbulence Kernel Fit")
            ofit_results,ofit_params,ofit_model_stars = self.ofit(stars,wcs,pointing,logger=logger)

        if self.do_sfit:
            logger.info("Step 2: Individual Star Turbulence Kernel Fit")
            sfit_results,sfit_params,sfit_model_stars = self.sfit(stars,wcs,pointing,logger=logger)

        logger.info("Step 3: Interpolation of Turbulence Kernels")

        return ofit_results,ofit_params,sfit_results,sfit_params

    def ofit(self,stars,wcs,pointing,logger=None):
        """Optical Fit step of the OptAtmo PSF

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # the following information is assumed to be present in the PSF objects
        # kwargs:  self.ofit_initvalues, self.ofit_bounds, self.ofit_double_zernike_terms, self.ofit_fix, self.ofit_optimizer
        #          self.ofit_nstars, self.ofit_shape_kwargs
        #          self.model
        # outout:  self.ofit_chiparam

        # setup the optical + constant atmosphere (ofit) parameters
        ofit_params = self._setup_ofit_params(self.ofit_initvalues,self.ofit_bounds,self.ofit_double_zernike_terms,self.ofit_fix)

        # add Zernike wavefront coefficients to stars, overwrite self.stars list
        new_stars = self._get_refwavefront(stars, logger)
        self.stars = new_stars

        # calculate moments for all Stars, currently options are hardcoded - TODO: put ofit_model_names in init...
        calc_moments_kwargs = {}
        calc_moments_kwargs['errors'] = True
        calc_moments_kwargs['third_order'] = True
        calc_moments_kwargs['fourth_order'] = False
        calc_moments_kwargs['radial'] = True
        self.ofit_moment_names = get_moment_names(**calc_moments_kwargs)
        calc_moments_kwargs['kernel'] = self.moment_kernel
        star_moments = self._calc_moments(stars=new_stars, logger=logger, **calc_moments_kwargs,addtostar=True)

        # select a smaller subset of stars, for use in the Optical fitting step
        select_ofit_stars_kwargs = {'nstars': self.ofit_nstars}
        ofit_stars = self._select_ofit_stars(new_stars, logger, **select_ofit_stars_kwargs)
        ofit_moments = self._get_moments(ofit_stars)

        # arguments to optimizer
        # least_squares defaults for ftol and xtol are 1.e-8
        ofit_kwargs = {'diff_step': 1.e-5, 'ftol': 1.e-4, 'xtol': 1.e-4}
        ofit_func_kwargs = {}  # TODO: things for chivec calculation

        # list of ndarrays with Chi2 and parameters for each optimization interation
        self.ofit_chiparam = []

        # perform ofit minimization, TODO: make this an option
        # add to yaml ofit_optimizer = 'iminuit'
        ofit_model = self.model

        if self.ofit_optimizer=='least_squares':

            ofit_results = least_squares(self._calc_ofit_residuals, ofit_params.getFloatingValues(), bounds=ofit_params.getFloatingBounds(), **ofit_func_kwargs,
                                         args=(ofit_params, ofit_stars, ofit_moments, ofit_model, logger, self.ofit_shape_kwargs))

            # fill parameter results into ofit_params
            ofit_params.setFloatingValues(ofit_results.x)

        elif self.ofit_optimizer=='iminuit':
            # setup MINUIT
            self.Minuit_args = (ofit_params, ofit_stars, ofit_moments, ofit_model, logger, self.ofit_shape_kwargs)  #there is no Minuit mechanism to pass args to the function
            gMinuit = Minuit(self._calc_ofit_chi2,ofit_params.getValues(),name=ofit_params.getNames())
            self._setup_minuit(gMinuit,ofit_params)

            # print out
            print(tab.tabulate(*gMinuit.params.to_table()))

            # do the fit
            gMinuit.migrad()

            # print results
            print(tab.tabulate(*gMinuit.params.to_table()))

            # get fit details from iminuit
            amin = gMinuit.fmin.fval
            edm = gMinuit.fmin.edm
            errdef = gMinuit.fmin.errordef
            nvpar = gMinuit.nfit
            nparx = gMinuit.npar
            icstat = int(gMinuit.fmin.is_valid) + 2*int(gMinuit.fmin.has_accurate_covar)
            dof = pow(19,2) - nvpar   #stamp_size is = 19

            mytxt = "amin = %.3f, edm = %.3f,   effdef = %.3f,   nvpar = %d,  nparx = %d, icstat = %d " % (amin,edm,errdef,nvpar,nparx,icstat)
            print('donutfit: ',mytxt)

            # get fit values and print errors
            ofit_params.setValues(gMinuit.values)
            print(gMinuit.errors)

            # save results...
            ofit_results = gMinuit


        # print some results
        logger.info("Optical Fit: fit results")
        print(ofit_results)
        ofit_params.print()

        # save results for output
        # self.ofit_param_values = ofit_params.getValues()

        # make model stars for all stars
        ofit_model_stars = self.make_modelstars(ofit_params,self.stars,self.model,logger=logger)

        # calculate moments for model stars, currently options are hardcoded, and moments are added to the modeL_stars
        calc_moments_kwargs = {}
        calc_moments_kwargs['errors'] = False
        calc_moments_kwargs['third_order'] = True
        calc_moments_kwargs['fourth_order'] = False
        calc_moments_kwargs['radial'] = True
        ofit_model_moments = self._calc_moments(ofit_model_stars,logger=logger,**calc_moments_kwargs,addtostar=True)

        return ofit_results,ofit_params,ofit_model_stars

    def sfit(self,stars,wcs,pointing,logger=None):
        """Single Star Fit step of the OptAtmo PSF

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # the following information is assumed to be present in the PSF objects
        # kwargs:  self.sfit_pixel_kwargs
        #          self.ofit_model_stars
        #          self.model
        # outout:  self.sfit_chiparam

        # output lists
        sfit_results_list = []
        sfit_params_list = []
        sfit_model_stars_list = []
        self.sfit_chiparam = []


        # loop over ALL stars, and fit an individual size,L0,g1,g2 using a pixel-based fit
        #for i in range(len(stars)):
        for i in range(len(stars)):

            data_star = stars[i]

            #if i == 100*int(i/100):
            print('Single star fit #',i)
            # build sfit parameters for this star
            ofit_model_star = self.ofit_model_stars[i]

            # TODO: get the ofit_params from the Fit of the data_stars
            sfit_params = self._setup_sfit_params(self.ofit_params,data_star,ofit_model_star) #use values from ofit, and get good starting values from the difference in e0,e1,e2,du,dv between data and ofit_model

            if self.sfit_optimizer=='least_squares':
                # fit the star
                # old # sfit_kwargs = {'method':'trf','diff_step': 1.e-5,'verbose':1,'ftol':1e-13,'xtol':1e-13, 'gtol':1e-13, 'jac':'2-point'}
                sfit_kwargs = {'method':'trf','verbose':1,'diff_step':1e-5,'ftol':1e-8,'xtol':1e-5, 'gtol':1e-8, 'jac':'2-point','x_scale':'jac'}  # Hardcoded now
                try:
                    sfit_result = least_squares(self._calc_sfit_residuals, sfit_params.getFloatingValues(), bounds=sfit_params.getFloatingBounds(), **sfit_kwargs,
                              args=(sfit_params, [data_star], self.model, logger, self.sfit_pixel_kwargs))
                except:
                    logger.warning("Single star fit failed")
                    sfit_result = None
                sfit_results_list.append(sfit_result)
                sfit_params_list.append(sfit_params)

            # do single star fit with iMinuit..
            elif self.sfit_optimizer=='iminuit':
                # setup MINUIT
                self.Minuit_args = (sfit_params, [data_star], self.model, logger, self.sfit_pixel_kwargs)  #there is no Minuit mechanism to pass args to the function
                gMinuit = Minuit(self._calc_sfit_chi2,sfit_params.getValues(),name=sfit_params.getNames())
                self._setup_minuit(gMinuit,sfit_params,tol=100.0)   #TODO: add option for this Tolerance

                # diagnostic output
                #print("Staring Parameters")
                #sfit_params.print()

                # do the fit
                gMinuit.migrad()

                # get fit details from iminuit
                amin = gMinuit.fmin.fval
                edm = gMinuit.fmin.edm
                errdef = gMinuit.fmin.errordef
                nvpar = gMinuit.nfit
                nparx = gMinuit.npar
                icstat = int(gMinuit.fmin.is_valid) + 2*int(gMinuit.fmin.has_accurate_covar)
                dof = pow(19,2) - nvpar   #stamp_size is = 19

                mytxt = "amin = %.3f, edm = %.3f,   effdef = %.3f,   nvpar = %d,  nparx = %d, icstat = %d " % (amin,edm,errdef,nvpar,nparx,icstat)
                #print('donutfit: ',mytxt)

                # get fit values and errors
                sfit_params.setValues(gMinuit.values)
                sfit_params.setErrors(gMinuit.errors)
                sfit_params_list.append(sfit_params)

                # save results tuyple
                results = (amin,edm,icstat,gMinuit.nfcn)
                sfit_results_list.append(results)



            # make a new model star with these parameters
            model_stars = self.make_modelstars(sfit_params,[data_star],self.model,logger=logger)
            sfit_model_stars_list.append(model_stars[0])

            # TODO: build a new data star with the sfit results in the Fit object

        # calculate moments for all sfit_model_stars_list
        calc_moments_kwargs = {}
        # TODO: add kernel type - or better get these settings from init
        calc_moments_kwargs['errors'] = False
        calc_moments_kwargs['third_order'] = True
        calc_moments_kwargs['fourth_order'] = False
        calc_moments_kwargs['radial'] = True
        sfit_model_moments = self._calc_moments(sfit_model_stars_list,logger=logger,**calc_moments_kwargs,addtostar=True)


        # return
        return sfit_results_list,sfit_params_list,sfit_model_stars_list

    def _chivec_vs_params(self,param_index,param_nbin,param_range,params,ofit_stars,ofit_moments,ofit_model,logger,ofit_shape_kwargs):
        """Calculate Optical fit residuals vector for a range of parameter values

        :param param_index:     Index of the parameter to varying
        :param param_nbin:      Number of points to for the varying parameter
        :param param_range:     A Tuple with the range of the parameter value to explore
        :param params:          A list of the free or fixed parameters of the Optical+UniformAtmosphere PSF
        :param stars:           A list of Stars to be used in the fit
        :param moments:         A np.ndarray with moments from data Stars
        :param model:           A Piff Model object encapsulating the Optical+UniformAtmosphere PSF
        :param logger:          A logger object for logging debug info. [default: None]
        :param kwargs:          A dictionary of other options
        """

### Optical Fit functions

    def _select_ofit_stars(self, stars, logger=None, nstars=500):
        """Select subset of stars for Optical+UniformAtmosphere fit

        :param stars:           A list of input Stars
        :param logger:          A logger object for logging debug info. [default: None]
        :param nstars:          Number of stars to select [default: 500]
        :return select_stars:   A list of output Stars
        """
        # sorts stars by flux and chooses largest nstar of them
        flux_array = self._get_flux(stars)
        sorted_list = np.argsort(flux_array)

        # set a flag in the properties of the selected stars
        for index in sorted_list[-nstars:]:
            stars[index].data.properties['is_ofit'] = True
        for index in sorted_list[0:-nstars]:
            stars[index].data.properties['is_ofit'] = False

        # form list of selected stars and return
        select_stars = [stars[index] for index in sorted_list[-nstars:]]
        return select_stars

    def _setup_ofit_params(self,ofit_initvalues,ofit_bounds,ofit_double_zernike_terms,ofit_fix=None):
        """ Setup the Optical fit parameters

        :param ofit_initvalues:
        :param ofit_bounds:
        :param ofit_double_zernike_terms:
        """
        ofit_params = Params()
        ofit_params.register('opt_size',initvalue=ofit_initvalues.get('opt_size',1.0),bounds=ofit_bounds.get('opt_size'))
        ofit_params.register('opt_L0',initvalue=ofit_initvalues.get('opt_L0',25.0),bounds=ofit_bounds.get('opt_L0',[5.0,100.0]))
        ofit_params.register('opt_g1',initvalue=ofit_initvalues.get('opt_g1',0.0),bounds=ofit_bounds.get('opt_g1'))
        ofit_params.register('opt_g2',initvalue=ofit_initvalues.get('opt_g2',0.0),bounds=ofit_bounds.get('opt_g2'))

        # add double Zernike coeffiencts
        for zf_pair in ofit_double_zernike_terms:
            iZ, nF = zf_pair
            for iF in range(1,nF+1):
                name = 'z%df%d' % (iZ,iF)
                ofit_params.register(name,initvalue=ofit_initvalues.get(name,0.0),bounds=ofit_bounds.get(name))

        # fix parameters as desired
        if ofit_fix:
            for name in ofit_fix:
                ofit_params.fix(name)

        return ofit_params

    def _calc_ofit_residuals(self,free_params,params,stars,moments,model,logger,kwargs):
        """Calculate Optical fit residuals

        :param free_params:     A List with the free parameters of the Optical+UniformAtmosphere PSF
        :param params:          A Params object with all parameters
        :param stars:           A list of Stars to be used in the fit
        :param moments:         A np.ndarray with moments from data Stars
        :param model:           A Piff Model object encapsulating the Optical+UniformAtmosphere PSF
        :param logger:          A logger object for logging debug info. [default: None]
        :param kwargs:          A dictionary of other options
        """
        # load floating params into params
        params.setFloatingValues(free_params)

        # make model stars, one for each ofit star
        model_stars = self.make_modelstars(params,stars,model,logger=logger)

        # calculate moments for model stars, currently options are hardcoded
        calc_moments_kwargs = {}
        calc_moments_kwargs['errors'] = False
        calc_moments_kwargs['third_order'] = True
        calc_moments_kwargs['fourth_order'] = False
        calc_moments_kwargs['radial'] = True
        model_moments = self._calc_moments(model_stars,logger=logger,**calc_moments_kwargs,addtostar=False)

        # calculate the chi vector - dimensionality and ordering given by [nstars,nparams]
        calc_chivec_kwargs = kwargs  #moment_list,weights,systerrors
        chivec = self._calc_chivec(moments,model_moments,logger=logger,**calc_chivec_kwargs)

        # info printout
        chi2 = np.sum(chivec*chivec)
        logger.info("Chi2 = %f" % (chi2))

        # save chi2,parameter values for each interation
        parvalues = params.getValues()
        chiparam = np.zeros(len(parvalues)+1)
        chiparam[0] = chi2
        chiparam[1:] = parvalues
        self.ofit_chiparam.append(chiparam)

        # a flattened vector is expected
        return chivec.flatten()

    def _calc_ofit_chi2(self,ofit_values):
        """Calculate chi2 for Minuit optimizer

        : param ofit_values     An ndarray of parameter values
        """

        # unpack saved quantities
        ofit_params,ofit_stars,ofit_moments,ofit_model,logger,kwargs = self.Minuit_args

        # set ofit_params to values input from Minuit, includes both Floating and Fixed parameters
        ofit_params.setValues(ofit_values)

        # for debugging...
        ofit_params.printChanges()

        # calculate residuals vector, reuse code!
        residuals = self._calc_ofit_residuals(ofit_params.getFloatingValues(),ofit_params,ofit_stars,ofit_moments,ofit_model,logger,kwargs)

        # Chi2
        chi2 = np.sum(residuals*residuals)
        return chi2


### Single Star Fit functions

    def _setup_sfit_params(self,ofit_params,data_star,ofit_model_star):
        """ Setup the Single Star fit parameters

        :param ofit_params:          Params object from Optical fit
        :param data_star:            Star to be fit
        :param ofit_model_stars:     Model from Optical fit
        :return sfit_params:         Params object appropriate for Single Star fit
        """
        sfit_params = Params()

        # copy parameters from Optical Fit, register these as fixed
        for name in ofit_params.getNames():
            sfit_params.register(name,initvalue=ofit_params.get(name),floating=False)

        # Starting values from difference in HSM moments in the data and the ofit model star
        init_flux = data_star.hsm[0]/ofit_model_star.hsm[0]
        init_size = (data_star.hsm[3]-ofit_model_star.hsm[3]) * 3.370   #scale factors found empirically in OptAtmo-test-newcode-part7.ipynb
        init_g1 = (data_star.hsm[4]-ofit_model_star.hsm[4]) * 3.689   #scale factors found empirically
        init_g2 = (data_star.hsm[5]-ofit_model_star.hsm[5]) * 3.689   #scale factors found empirically
        init_du = (data_star.hsm[1]-ofit_model_star.hsm[1])
        init_dv = (data_star.hsm[2]-ofit_model_star.hsm[2])


        # add Atmo parameters
        sfit_params.register('atmo_size',initvalue=init_size,bounds=[-0.25,0.25],initerror=0.01)
        sfit_params.register('atmo_g1',initvalue=init_g1,bounds=[-1.00,1.00],initerror=0.005)
        sfit_params.register('atmo_g2',initvalue=init_g2,bounds=[-1.00,1.00],initerror=0.005)
        twopixel = 0.263*2.0    # need a range of 2 pixels to allow for Coma
        sfit_params.register('du',initvalue=init_du,bounds=[-twopixel,twopixel],initerror=0.005)
        sfit_params.register('dv',initvalue=init_dv,bounds=[-twopixel,twopixel],initerror=0.005)
        sfit_params.register('flux',initvalue=init_flux,bounds=[init_flux*0.9,init_flux*1.1],initerror=np.sqrt(init_flux))


        return sfit_params

    def _calc_sfit_residuals(self,free_params,params,stars,model,logger,kwargs):
        """Calculate Optical fit residuals. Note: this is designed to work for a
        list of input stars or with a list of a single star

        :param free_params:     A List with the free parameters of the Optical+UniformAtmosphere PSF
        :param params:          A Params object with all parameters
        :param stars:           A list of Stars to be used in the fit
        :param model:           A Piff Model object encapsulating the Optical+UniformAtmosphere PSF
        :param logger:          A logger object for logging debug info. [default: None]
        :param kwargs:          A dictionary of other options
        """
        # load floating params into params
        params.setFloatingValues(free_params)

        # make model stars, one for each star
        model_stars = self.make_modelstars(params,stars,model,logger=logger)

        # calculate the chi vector for each pixel - dimensionality and ordering given by [nstars,npixels]
        calc_chivec_kwargs = kwargs  # TODO: add radius of pixels to use in calculating the chivec

        # calculate the chi vector per pixel
        nstar = len(stars)

        # calculate the number of pixels, using a mask based on the opt_size
        # TODO: this is done for every iteration - do only once per star!

        usemask = False
        if usemask:
            _, _, u, v = stars[0].data.getDataVector(include_zero_weight=True)
            size =params.get('opt_size') # use the optics size parameter, which is roughly the FWHM in arcsec
            rsq = (u**2 + v**2)/(size/2.)**2   # so rsq = 1 is within the HWHM
            rsqmax = kwargs['pixel_radius']**2   # in units of HWHM
            mask = (rsq < rsqmax )
            npixel = int(np.sum(mask))
        else:
            _, _, u, v = stars[0].data.getDataVector(include_zero_weight=True)
            npixel = u.shape[0]
        #npixel = stars[0].image.array.flatten().shape[0]  # assumes the first star has a full NxN image
        chivec = np.zeros((nstar,npixel))

        # loop over stars here...
        for i,star in enumerate(stars):
            data, weight, u, v = star.data.getDataVector(include_zero_weight=True)
            model_data = model_stars[i].image.array.flatten()

            # normalize model_data to data, with a simple sum, dont need if we vary a flux parameter
            # model_data_normed = model_data * np.sum(data)/np.sum(model_data)
            # chivec[i,:] = (data[mask]-model_data_normed[mask])*np.sqrt(weight[mask])
            # calculate the pull or chi vector
            if usemask:
                chivec[i,:] = (data[mask]-model_data[mask])*np.sqrt(weight[mask])
            else:
                chivec[i,:] = (data-model_data)*np.sqrt(weight)

        # both data and data_model are float32, but chivec needs to be float64 for least_squares to work correctly!
        chivec = chivec.astype(np.float64)

        # print which parameters changed, TODO: only for debug mode...
        #params.printChanges()

        # info printout
        chi2 = np.sum(chivec*chivec)
        #logger.info("Chi2 = %f" % (chi2))

        # save chi2,parameter values for each interation
        parvalues = params.getValues()
        chiparam = np.zeros(len(parvalues)+1)
        chiparam[0] = chi2
        chiparam[1:] = parvalues
        self.sfit_chiparam.append(chiparam)

        # debug printout
        #print(params.getValues())
        #print("Chi2 = ",chi2)

        # a flattened vector is expected
        return chivec.flatten()


    def _calc_sfit_chi2(self,sfit_values):
        """Calculate chi2 for Minuit optimizer for Single Star fit

        : param sfit_values     An ndarray of parameter values
        """

        # unpack saved quantities
        params,stars,model,logger,kwargs = self.Minuit_args

        # set ofit_params to values input from Minuit, includes both Floating and Fixed parameters
        params.setValues(sfit_values)

        # calculate residuals vector, reuse code!
        residuals = self._calc_sfit_residuals(params.getFloatingValues(),params,stars,model,logger,kwargs)

        # Chi2
        chi2 = np.sum(residuals*residuals)
        return chi2

###   Utility Functions

    def _setup_minuit(self,gMinuit,params,tol=1.0,print_level=1,strategy=0):
        """Setup for iMinuit fitter

        :param gMinuit:         An iMinuit object
        :param params:          A Params object with the free or fixed parameters of the fitting function
        :param tol:             Tolerance level [default: 1.0]
        :param print_level:     Print level [default: 1]
        :param strategy:        Strategy flag [default: 0]
        """

        # setup gMinuit object for a least squares fit
        gMinuit.strategy = strategy
        gMinuit.errordef = Minuit.LEAST_SQUARES
        gMinuit.print_level = print_level
        gMinuit.tol = tol  # 0.1 is the default, increase to speed fitting at the cost of not being exactly at the minimum

        # setup parameters from params Object
        for ipar,aname in enumerate(params.getNames()):
            gMinuit.values[aname] = params.get(aname)
            lo,hi = params.getBounds(aname)
            if lo!=-np.inf or hi!=np.inf :
                gMinuit.limits[aname] = (lo,hi)
            gMinuit.errors[aname] = params.getError(aname)
            if params.isFloat(aname):
                gMinuit.fixed[aname] = False
            else:
                gMinuit.fixed[aname] = True

        return

    def _get_minuit_results(self,gMinuit):
        """Get dictionary of results from iMinuit fitter

        :param gMinuit:         An iMinuit object
        :return retdict:        A dictionary of results from iMinuit
        """
        # get fit details from iminuit
        amin = gMinuit.fmin.fval
        edm = gMinuit.fmin.edm
        errdef = gMinuit.fmin.errordef
        nvpar = gMinuit.nfit
        nparx = gMinuit.npar
        icstat = int(gMinuit.fmin.is_valid) + 2*int(gMinuit.fmin.has_accurate_covar)
        dof = pow(19,2) - nvpar   #stamp_size is = 19

        mytxt = "amin = %.3f, edm = %.3f,   effdef = %.3f,   nvpar = %d,  nparx = %d, icstat = %d " % (amin,edm,errdef,nvpar,nparx,icstat)
        print('donutfit: ',mytxt)
        return mytxt



    def make_modelstars(self,params,stars,model,logger=None):
        """Make model stars for each of the input stars, given the parameters params

        :param params:          A Params object with the free or fixed parameters of the Optical+UniformAtmosphere PSF
        :param stars:           A list of Stars to be used in the fit
        :param model:           A Piff Model object with the Optical+UniformAtmosphere PSF
        :param logger:          A logger object for logging debug info. [default: None]
        :return model_stars:    A list of Model Stars
        """

        # Draw the model stars
        model_stars = []
        for i,astar in enumerate(stars):

            # assume that opt_XXX parameters are always present. i
            model_kwargs = {}
            model_kwargs['r0'] = 0.15/params.get('opt_size')  #define size == 0.15/r0
            model_kwargs['L0'] = params.get('opt_L0')
            model_kwargs['g1'] = params.get('opt_g1')
            model_kwargs['g2'] = params.get('opt_g2')

            # if atmo_XXX parameters are present then include those as well.
            if params.hasparam('atmo_size'):
                model_kwargs['r0'] = 0.15/(params.get('opt_size')+params.get('atmo_size'))
            if params.hasparam('atmo_g1'):
                model_kwargs['g1'] = params.get('opt_g1')+params.get('atmo_g1')
            if params.hasparam('atmo_g2'):
                model_kwargs['g2'] = params.get('opt_g2')+params.get('atmo_g2')

            # if atmo_XXX_NN parameters are present then include those as well.
            if params.hasparam('atmo_size_%d' % (i)):
                model_kwargs['r0'] = 0.15/(params.get('opt_size')+params.get('atmo_size_%d' % (i)))
            if params.hasparam('atmo_g1_%d' % (i)):
                model_kwargs['g1'] = params.get('opt_g1')+params.get('atmo_g1_%d' % (i))
            if params.hasparam('atmo_g2_%d' % (i)):
                model_kwargs['g2'] = params.get('opt_g2')+params.get('atmo_g2_%d' % (i))


            # retrieve the reference wavefront values from the input star
            aArr = astar.data.properties['wavefront'].copy()  #don't change it inside the star please!

            # Field position [arcsec] on sky.
            u = astar.data.properties['u'] / self.fov_radius
            v = astar.data.properties['v'] / self.fov_radius

            # add in changes to aberrations
            for j,zf_pair in enumerate(self.ofit_double_zernike_terms):

                iZ,nF = zf_pair
                # NOTE: currently hardcoded that only 1,3,6 or 10 Focal Plane Zernike terms are allowed
                # using Noll Convention for Focal Plane Zernike terms, see https://spie.org/etop/1997/382_1.pdf for trigonometry
                if nF not in [1,3,6,10]:
                    raise ValueError("Incorrect specification of Double Zernike terms: %d" % (self.ofit_double_zernike_terms))

                if nF>=1:
                    aArr[iZ] += params.get('z%df%d' % (iZ,1))
                if nF>=3:
                    aArr[iZ] += ( u * params.get('z%df%d' % (iZ,2)) +
                                  v * params.get('z%df%d' % (iZ,3)) )
                if nF>=6:
                    usq = u*u
                    vsq = v*v
                    aArr[iZ] += ( (2*usq + 2*vsq - 1) * params.get('z%df%d' % (iZ,4)) +
                                  (2*u*v) * params.get('z%df%d' % (iZ,5)) +
                                  (usq - vsq) * params.get('z%df%d' % (iZ,6)) )
                if nF>=10:
                    ucub = usq*u
                    vcub = vsq*v
                    aArr[iZ] += ( (3*vcub + 3*usq*v - 2*v) * params.get('z%df%d' % (iZ,7)) +
                                  (3*ucub + 3*vsq*u - 2*u) * params.get('z%df%d' % (iZ,8)) +
                                  (-vcub + 3*usq*v) * params.get('z%df%d' % (iZ,9)) +
                                  (ucub - 3*vsq*u) * params.get('z%df%d' % (iZ,10)) )

            # put aberrations in the model_kwargs
            model_kwargs['zernike_coeff'] = aArr

            # build the Profile from our model.  This is the Galsim object with the PSF
            prof = model.getProfile(**model_kwargs)

            # set star's fit center,flux to value in params object, if present.
            # model.drawProfile uses the values in the StarFit to set center and flux
            # Parameter names use index in star list for pixel based ofit
            if params.hasparam('du_%d' % (i)):
                astar.fit.center = (params.get('du_%d',(i)),params.get('dv_%d',(i)))
            if params.hasparam('flux_%d' % (i)):
                astar.fit.flux = params.get('flux_%d',(i))

            # set star's fit center,flux to value in params, if present.
            if params.hasparam('du'):
                astar.fit.center = (params.get('du'),params.get('dv'))
            if params.hasparam('flux'):
                astar.fit.flux = params.get('flux')


            # draw a model star, using the data star as a template for the image, with location, wcs, and weight mask
            model_stars.append(model.drawProfile(astar,prof,params.getValues()[0:4]))  #TODO: hardcoded to only store size,L0,g1,g2 in the StarFit

        return model_stars

    def _get_refwavefront(self,stars,logger=None):
        """Get Zernike wavefront coefficients, add to Star's properties.

        :param stars:           A list of input Stars
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # make the Wavefront object, with kwargs
        wfobj = Wavefront(self.wavefront_kwargs,logger=logger)

        # fill Wavefront in star.data.properties
        new_stars = wfobj.fillWavefront(stars,logger=logger)

        return new_stars

    def _calc_moments(self,stars,kernel='Gaussian',third_order=False,fourth_order=False, radial=False,errors=False,addtostar=False,logger=None):
        """Calculate moments, add to Star's properties, and return as an array

        :param stars:           A list of input Stars
        :param kernel:          Kernel type: Gaussian or VonKarman [default:Gaussian]
        :param third_order:     Return the 3rd order moments? [default: False]
        :param fourth_order:    Return the 4th order moments? [default: False]
        :param radial:          Return the higher order radial moments? [default: False]
        :param errors:          Return the variance estimates of other returned values? [default: False]
        :param addtostar:       Add the moments to the Stars's properties? [default False]
        :param logger:          A logger object for logging debug info. [default: None]
        :return moments:        An ndarray of moments for all input Stars, [istar,imoment]
        """

        nstars = len(stars)

        # Loop over stars, calculating the moments
        for i, astar in enumerate(stars):
            # in first iteration find the size of the tuple returned, and use
            # to build ndarray
            if i == 0:
                moms = calculate_moments(astar,kernel=kernel,third_order=third_order,fourth_order=fourth_order,
                                             radial=radial,errors=errors,logger=logger)
                moments = np.zeros((nstars,len(moms)))
                moments[i,:] = moms
            else:
                moments[i,:] = calculate_moments(astar,kernel=kernel,third_order=third_order,fourth_order=fourth_order,
                                                     radial=radial,errors=errors,logger=logger)

            if addtostar:
                astar.data.properties['moments'] = moments[i,:]

        return moments

    def _calc_chivec(self,moments,model_moments,logger=None,moment_list=['e0','e1','e2'],weights=None,systerrors=None):
        """Calculate Chi Vector for Star data moments and model moments

        :param moments:         An ndarray of data moments [istar,imoment]
        :param model_moments:   An ndarray of model moments
        :param logger:          A logger object for logging debug info. [default: None]
        :param moment_list:     A list of the moment indicies to use in the Chi vector [default: ['e0','e1','e2']]
        :param weights:         A list of the weighting factors to use in the Chi vector, ordered as in moment_list [default: None]
        :param systerrors:      A list of the systematic errors to add in quadrature with the measured errors in each Chi term, ordered as in moment_list [default: None]
        :return chivec:         An ndarray of Chi = (data-model)/error, [istar,imoment]
        """

        # build moment_indices and error_indices from moment_list, since not all moments are used in chivec
        # TODO: consider if structured arrays would be a better way to handle this...
        moment_indices = [ self.ofit_moment_names.index(name) for name in moment_list]
        error_indices = [ self.ofit_moment_names.index(name+'_err') for name in moment_list]

        # extract elements and calculate chivec array ordered as: [istar,imoment]
        d = moments[:,moment_indices]
        m = model_moments[:,moment_indices]
        e2 = moments[:,error_indices]   #Using variance from data

        # add systematic errors to moment errors
        if systerrors != 'None' and systerrors != None :
            s = np.array(systerrors)
            e2 = np.sqrt(e2 + s*s)

        # calculate chivec
        chivec = (d-m)/np.sqrt(e2)

        # apply weights vector to chivec
        if weights != 'None' and weights != None :
            chivec = chivec * np.array(weights)

        # return array with dimension [imoments,istar]
        return chivec

    def _get_moments(self,stars):
        """Get moments from Star's properties

        :param stars:           A list of input Stars
        :return moments_array:  An ndarray with Star moments in it [imoments,istar]
        """

        # get size of moments array from 1st star
        nmoments = stars[0].data.properties['moments'].shape[0]
        nstars = len(stars)

        moments_array =  np.zeros((nstars,nmoments))
        for i,astar in enumerate(stars):
            moments_array[i,:] = astar.data.properties['moments']

        return moments_array

    # add to util.py?
    def _get_flux(self,stars):
        """Get flux from Star's properties

        :param stars:           A list of input Stars
        :return flux_array:     An ndarray with Star flux in it
        """
        # weighted flux is always stored as the first element of the moments
        # property
        ind_flux = 0

        flux_array = np.zeros(len(stars))
        for i, astar in enumerate(stars):
            flux_array[i] = astar.data.properties['moments'][ind_flux]

        return flux_array


    # TODO: these methods need to be revised for OptAtmo_PSF
    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        stars = self.interp.interpolateList(stars)
        for star in stars:
            self.model.normalize(star)
        return stars

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        star = self.interp.interpolate(star)
        self.model.normalize(star)
        return star

    def _drawStar(self, star, copy_image=True, center=None):
        return self.model.draw(star, copy_image=copy_image, center=center)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        logger = galsim.config.LoggerWrapper(logger)

        # TODO: write out the fit param vector, information about the fit (chi2, chivec, quality etc..), model stars, the refwf kwargs, and any other information need to reproduce the fit
        # and then also the individual stars fit results and GP too...
        # the optical model and GP interpreter are writen out by their own write methods....

        # ofit chi2,params per iteration
        fits.write_table(self.ofit_chiparam,extname=extname + '_ofit_chiparam')
        logger.debug("Wrote the ofit chiparam table to extension %s",extname + '_ofit_chiparam')

        # ofit param values
        write_kwargs(fits, extname + '_ofit_param_values', self.ofit_param_values)
        logger.debug("Wrote the ofit param values to extension %s",extname + '_ofit_param_values')

        # write out ofit model stars
        Star.write(self.ofit_model_stars, fits, extname=extname + '_ofit_model_stars')
        logger.info("Wrote the ofit model stars to extname %s", extname + 'ofit_model_stars')

        # write out model,interp and outliers
        self.model.write(fits, extname + '_model')
        logger.debug("Wrote the PSF model to extension %s",extname + '_model')
        self.interp.write(fits, extname + '_interp')
        logger.debug("Wrote the PSF interp to extension %s",extname + '_interp')
        if self.outliers:
            self.outliers.write(fits, extname + '_outliers')
            logger.debug("Wrote the PSF outliers to extension %s",extname + '_outliers')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        chisq_dict = read_kwargs(fits, extname + '_chisq')
        for key in chisq_dict:
            setattr(self, key, chisq_dict[key])
        self.model = Model.read(fits, extname + '_model')
        self.interp = Interp.read(fits, extname + '_interp')
        if extname + '_outliers' in fits:
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            self.outliers = None
