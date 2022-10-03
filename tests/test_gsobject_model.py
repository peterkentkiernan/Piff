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

from __future__ import print_function
import galsim
import treecorr
import treegp
import piff
import numpy as np
import os
import fitsio

from piff_test_helper import timer

fiducial_kolmogorov = galsim.Kolmogorov(half_light_radius=1.0)
fiducial_gaussian = galsim.Gaussian(half_light_radius=1.0)
fiducial_moffat = galsim.Moffat(half_light_radius=1.0, beta=3.0)

def make_data(gsobject, scale, g1, g2, u0, v0, flux, noise=0., pix_scale=1., fpu=0., fpv=0.,
              nside=32, nom_u0=0., nom_v0=0., rng=None, include_pixel=True):
    """Make a Star instance filled with a Kolmogorov profile

    :param gsobject     The fiducial gsobject profile to use.
    :param scale:       The scale to apply to the gsobject.
    :param g1, g2:      The shear to apply to the gsobject.
    :param u0, v0:      The sub-pixel offset to apply.
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param pix_scale:   pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    :param include_pixel:  Include integration over pixel.  [default: True]
    """
    k = gsobject.withFlux(flux).dilate(scale).shear(g1=g1, g2=g2).shift(u0, v0)
    if noise == 0.:
        var = 1.e-6
    else:
        var = noise**2
    weight = galsim.Image(nside, nside, dtype=float, init_value=1./var, scale=pix_scale)
    star = piff.Star.makeTarget(x=nside/2+nom_u0/pix_scale, y=nside/2+nom_v0/pix_scale,
                                u=fpu, v=fpv, scale=pix_scale, stamp_size=nside, weight=weight)
    star.image.setOrigin(0,0)
    star.weight.setOrigin(0,0)
    method = 'auto' if include_pixel else 'no_pixel'
    k.drawImage(star.image, method=method,
                offset=galsim.PositionD(nom_u0/pix_scale, nom_v0/pix_scale), use_true_center=False)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


@timer
def test_simple():
    """Initial simple test of Gaussian, Kolmogorov, and Moffat PSFs.
    """
    # Here is the true PSF
    scale = 1.3
    g1 = 0.23
    g2 = -0.17
    du = 0.1
    dv = 0.4
    for fiducial in [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]:
        print()
        print("fiducial = ", fiducial)
        print()
        psf = fiducial.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64, 64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        fiducial_star = piff.Star(stardata, None)

        # First try fastfit.
        print('Fast fit')
        model = piff.GSObjectModel(fiducial, fastfit=True, include_pixel=False)
        fit = model.fit(model.initialize(fiducial_star)).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-4)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-7)

        # Now try fastfit=False.
        print('Slow fit')
        model = piff.GSObjectModel(fiducial, fastfit=False, include_pixel=False)
        fit = model.fit(model.initialize(fiducial_star)).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-6)

        # Now test running it via the config parser
        config = {
            'model' : {
                'type' : 'GSObjectModel',
                'gsobj': repr(fiducial),
                'include_pixel': False
            }
        }
        if __name__ == '__main__':
            logger = piff.config.setup_logger(verbose=3)
        else:
            logger = piff.config.setup_logger(verbose=1)
        model = piff.Model.process(config['model'], logger)
        fit = model.fit(model.initialize(fiducial_star)).fit

        # Same tests.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-6)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__

        # Finally, we should also test with pixel convolution included.  This really only makes
        # sense for fastfit=False, since HSM FindAdaptiveMom doesn't account for the pixel shape
        # in its measurements.

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        psf.drawImage(image, method='auto')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        fiducial_star = piff.Star(stardata, None)


        print('Slow fit, pixel convolution included.')
        model = piff.GSObjectModel(fiducial, fastfit=False, include_pixel=True)
        star = model.initialize(fiducial_star)
        star = model.fit(star, fastfit=True)  # Get better results with one round of fastfit.
        # Use a no op convert_func, just to touch that branch in the code.
        convert_func = lambda prof: prof
        fit = model.fit(star, convert_func=convert_func).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # Accuracy goals are a bit looser here since it's harder to fit with the pixel involved.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)


@timer
def test_center():
    """Fit with centroid free and PSF center constrained to an initially mis-registered PSF.
    """
    influx = 150.
    scale = 2.0
    u0, v0 = 0.6, -0.4
    g1, g2 = 0.1, 0.2
    for fiducial in [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]:
        print()
        print("fiducial = ", fiducial)
        print()
        s = make_data(fiducial, scale, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)

        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        star = mod.initialize(s)
        print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
        for i in range(3):
            star = mod.fit(star)
            star = mod.reflux(star)
            print('Flux, ctr, chisq after fit {:d}:'.format(i),
                  star.fit.flux, star.fit.center, star.fit.chisq)
            np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=8)
            np.testing.assert_allclose(star.fit.center[0], u0)
            np.testing.assert_allclose(star.fit.center[1], v0)

        # Residual image when done should be dominated by structure off the edge of the fitted
        # region.
        mask = star.weight.array > 0
        # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
        star2 = mod.draw(star)
        print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
        print('max image abs value = ',np.max(np.abs(s.image.array)))
        peak = np.max(np.abs(s.image.array[mask]))
        np.testing.assert_almost_equal(star2.image.array[mask]/peak, s.image.array[mask]/peak,
                                       decimal=8)

        # Measured centroid of PSF model should be close to 0,0
        star3 = mod.draw(star.withFlux(influx, (0,0)))
        flux, cenx, ceny, sigma, e1, e2, flag = star3.hsm
        print('HSM measurements: ',flux, cenx, ceny, sigma, g1, g2, flag)
        np.testing.assert_allclose(cenx, 0, atol=1.e-4)
        np.testing.assert_allclose(ceny, 0, atol=1.e-4)
        np.testing.assert_allclose(e1, g1, rtol=1.e-4)
        np.testing.assert_allclose(e2, g2, rtol=1.e-4)

        # test copy_image
        star_copy = mod.draw(star, copy_image=True)
        star_nocopy = mod.draw(star, copy_image=False)
        star.image.array[0,0] = 132435
        assert star_nocopy.image.array[0,0] == star.image.array[0,0]
        assert star_copy.image.array[0,0] != star.image.array[0,0]
        assert star_copy.image.array[1,1] == star.image.array[1,1]

@timer
def test_uncentered():
    """Fit with centroid shift included in the PSF model.  (I.e. centered=False)
    """
    influx = 150.
    scale = 2.0
    u0, v0 = 0.6, -0.4
    g1, g2 = 0.1, 0.2
    for fiducial in [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]:
        print()
        print("fiducial = ", fiducial)
        print()
        s = make_data(fiducial, scale, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)

        mod = piff.GSObjectModel(fiducial, include_pixel=False, centered=False)
        star = mod.initialize(s)
        print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
        for i in range(3):
            star = mod.fit(star)
            star = mod.reflux(star)
            print('Flux, ctr, chisq after fit {:d}:'.format(i),
                  star.fit.flux, star.fit.center, star.fit.chisq)
            np.testing.assert_allclose(star.fit.flux, influx)
            np.testing.assert_allclose(star.fit.center[0], 0)
            np.testing.assert_allclose(star.fit.center[1], 0)

        # Residual image when done should be dominated by structure off the edge of the fitted
        # region.
        mask = star.weight.array > 0
        # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
        star2 = mod.draw(star)
        print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
        print('max image abs value = ',np.max(np.abs(s.image.array)))
        peak = np.max(np.abs(s.image.array[mask]))
        np.testing.assert_almost_equal(star2.image.array[mask]/peak, s.image.array[mask]/peak,
                                       decimal=8)

        # Measured centroid of PSF model should be close to u0, v0
        star3 = mod.draw(star.withFlux(influx, (0,0)))
        flux, cenx, ceny, sigma, e1, e2, flag = star3.hsm
        print('HSM measurements: ',flux, cenx, ceny, sigma, g1, g2, flag)
        np.testing.assert_allclose(cenx, u0, rtol=1.e-4)
        np.testing.assert_allclose(ceny, v0, rtol=1.e-4)
        np.testing.assert_allclose(e1, g1, rtol=1.e-4)
        np.testing.assert_allclose(e2, g2, rtol=1.e-4)


@timer
def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    influx = 150.
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
        niter = 3
        npos = 10
    else:
        fiducial_list = [fiducial_moffat]
        niter = 1  # Not actually any need for interating in this case.
        npos = 4
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        g1 = g2 = u0 = v0 = 0.0

        # Interpolator will be simple mean
        interp = piff.Polynomial(order=0)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,npos)
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            for v in positions:
                s = make_data(fiducial, 1.0, g1, g2, u0, v0, influx,
                              noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng, include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)
        s0 = mod.initialize(s0)

        # Polynomial doesn't need this, but it should work nonetheless.
        interp.initialize(stars)

        # Iterate solution using interpolator
        for iteration in range(niter):
            # Refit PSFs star by star:
            for i,s in enumerate(stars):
                stars[i] = mod.fit(s)
            # Run the interpolator
            interp.solve(stars)
            # Install interpolator solution into each
            # star, recalculate flux, report chisq
            chisq = 0.
            dof = 0
            for i,s in enumerate(stars):
                s = interp.interpolate(s)
                s = mod.reflux(s)
                chisq += s.fit.chisq
                dof += s.fit.dof
                stars[i] = s
            print('iteration',iteration,'chisq=',chisq, 'dof=',dof)

        # Now use the interpolator to produce a noiseless rendering
        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=3)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=3)


@timer
def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        g1 = g2 = u0 = v0 = 0.0

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        np_rng = np.random.RandomState(1234)
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                s = make_data(fiducial, 1.0, g1, g2, u0, v0, influx,
                              noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng, include_pixel=False)
                s = mod.initialize(s)
                # Kill 10% of each star's pixels
                bad = np_rng.rand(*s.image.array.shape) < 0.1
                s.weight.array[bad] = 0.
                s.image.array[bad] = -999.
                s = mod.reflux(s, fit_center=False) # Start with a sensible flux
                stars.append(s)

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)
        s0 = mod.initialize(s0)

        interp = piff.Polynomial(order=0)
        interp.initialize(stars)

        oldchisq = 0.
        # Iterate solution using interpolator
        for iteration in range(40):
            # Refit PSFs star by star:
            for i,s in enumerate(stars):
                stars[i] = mod.fit(s)
            # Run the interpolator
            interp.solve(stars)
            # Install interpolator solution into each
            # star, recalculate flux, report chisq
            chisq = 0.
            dof = 0
            for i,s in enumerate(stars):
                s = interp.interpolate(s)
                s = mod.reflux(s)
                chisq += s.fit.chisq
                dof += s.fit.dof
                stars[i] = s
                ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
            print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
            if oldchisq>0 and chisq<oldchisq and oldchisq-chisq < dof/10.:
                break
            else:
                oldchisq = chisq

        # Now use the interpolator to produce a noiseless rendering
        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
        # Less than 2 dp of accuracy here!
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=3)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=3)


@timer
def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)

        # Interpolator will be linear
        interp = piff.Polynomial(order=1)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            # Put gradient in pixel size
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                # spatially-varying fwhm, g1, g2.
                s = make_data(fiducial, 1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v, influx,
                                         noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng,
                                         include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(4, 4)
        # for star, ax in zip(stars, axes.ravel()):
        #     ax.imshow(star.data.image.array)
        # plt.show()

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, 0., 0., 0., 0., influx, pix_scale=0.5, include_pixel=False)
        s0 = mod.initialize(s0)

        # Polynomial doesn't need this, but it should work nonetheless.
        interp.initialize(stars)

        oldchisq = 0.
        # Iterate solution using interpolator
        for iteration in range(40):
            # Refit PSFs star by star:
            for i,s in enumerate(stars):
                stars[i] = mod.fit(s)
            # Run the interpolator
            interp.solve(stars)
            # Install interpolator solution into each
            # star, recalculate flux, report chisq
            chisq = 0.
            dof = 0
            for i,s in enumerate(stars):
                s = interp.interpolate(s)
                s = mod.reflux(s)
                chisq += s.fit.chisq
                dof += s.fit.dof
                stars[i] = s
                ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
            print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
            if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
                break
            else:
                oldchisq = chisq

        for i, s in enumerate(stars):
            print(i, s.fit.center)

        # Now use the interpolator to produce a noiseless rendering
        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


@timer
def test_gradient_center():
    """Next: fit spatially-varying PSF, with spatially-varying centers to multiple images.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)

        # Interpolator will be linear
        interp = piff.Polynomial(order=1)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            # Put gradient in pixel size
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                # spatially-varying fwhm, g1, g2.
                s = make_data(fiducial, 1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v,
                              influx, noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng,
                              include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(4, 4)
        # for star, ax in zip(stars, axes.ravel()):
        #     ax.imshow(star.data.image.array)
        # plt.show()

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, 0., 0., 0., 0., influx, pix_scale=0.5, include_pixel=False)
        s0 = mod.initialize(s0)

        # Polynomial doesn't need this, but it should work nonetheless.
        interp.initialize(stars)

        oldchisq = 0.
        # Iterate solution using interpolator
        for iteration in range(40):
            # Refit PSFs star by star:
            for i,s in enumerate(stars):
                stars[i] = mod.fit(s)
            # Run the interpolator
            interp.solve(stars)
            # Install interpolator solution into each
            # star, recalculate flux, report chisq
            chisq = 0.
            dof = 0
            for i,s in enumerate(stars):
                s = interp.interpolate(s)
                s = mod.reflux(s)
                chisq += s.fit.chisq
                dof += s.fit.dof
                stars[i] = s
                ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
            print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
            if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
                break
            else:
                oldchisq = chisq

        for i, s in enumerate(stars):
            print(i, s.fit.center, s.fit.params[0:2])

        # Now use the interpolator to produce a noiseless rendering
        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
        # Less than 2 dp of accuracy here!
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


@timer
def test_direct():
    """ Simple test for directly instantiated Gaussian, Kolmogorov, and Moffat without going through
    GSObjectModel explicitly.
    """
    # Here is the true PSF
    scale = 1.3
    g1 = 0.23
    g2 = -0.17
    du = 0.1
    dv = 0.4

    gsobjs = [galsim.Gaussian(sigma=1.0),
              galsim.Kolmogorov(half_light_radius=1.0),
              galsim.Moffat(half_light_radius=1.0, beta=3.0),
              galsim.Moffat(half_light_radius=1.0, beta=2.5, trunc=3.0)]

    models = [piff.Gaussian(fastfit=True, include_pixel=False),
              piff.Kolmogorov(fastfit=True, include_pixel=False),
              piff.Moffat(fastfit=True, beta=3.0, include_pixel=False),
              piff.Moffat(fastfit=True, beta=2.5, trunc=3.0, include_pixel=False)]

    for gsobj, model in zip(gsobjs, models):
        print()
        print("gsobj = ", gsobj)
        print()
        psf = gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        star = piff.Star(stardata, None)
        star = model.initialize(star)

        # First try fastfit.
        print('Fast fit')
        fit = model.fit(star).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        # These tests are more strict above.  The truncated Moffat included here but not there
        # doesn't work quite as well.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-4)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_direct_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__

    # repeat with fastfit=False

    models = [piff.Gaussian(fastfit=False, include_pixel=False),
              piff.Kolmogorov(fastfit=False, include_pixel=False),
              piff.Moffat(fastfit=False, beta=3.0, include_pixel=False),
              piff.Moffat(fastfit=False, beta=2.5, trunc=3.0, include_pixel=False)]

    for gsobj, model in zip(gsobjs, models):
        print()
        print("gsobj = ", gsobj)
        print()
        psf = gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        star = piff.Star(stardata, None)
        star = model.initialize(star)

        print('Slow fit')
        fit = model.fit(star).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-5)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_direct_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__

@timer
def test_var():
    """Check that the variance estimate in params_var is sane.
    """
    # Here is the true PSF
    scale = 1.3
    g1 = 0.23
    g2 = -0.17
    du = 0.1
    dv = 0.4
    flux = 500
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    noise = 0.2

    gsobjs = [galsim.Gaussian(sigma=1.0),
              galsim.Kolmogorov(half_light_radius=1.0),
              galsim.Moffat(half_light_radius=1.0, beta=3.0),
              galsim.Moffat(half_light_radius=1.0, beta=2.5, trunc=3.0)]

    # Mix of centered = True/False,
    #        fastfit = True/False,
    #        include_pixel = True/False
    models = [piff.Gaussian(fastfit=False, include_pixel=False, centered=False),
              piff.Kolmogorov(fastfit=True, include_pixel=True, centered=False),
              piff.Moffat(fastfit=False, beta=4.8, include_pixel=True, centered=True),
              piff.Moffat(fastfit=True, beta=2.5, trunc=3.0, include_pixel=False, centered=True)]

    names = ['Gaussian',
             'Kolmogorov',
             'Moffat3',
             'Moffat2.5']

    for gsobj, model, name in zip(gsobjs, models, names):
        print()
        print("gsobj = ", gsobj)
        print()
        psf = gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv).withFlux(flux)
        image = psf.drawImage(nx=64, ny=64, wcs=wcs, method='no_pixel')
        weight = image.copy()
        weight.fill(1/noise**2)
        # Save this one without noise.

        image1 = image.copy()
        image1.addNoise(galsim.GaussianNoise(sigma=noise))

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center, weight)
        star = piff.Star(stardata, None)
        star = model.initialize(star)
        fit = model.fit(star).fit

        file_name = 'input/test_%s_var.npz'%name
        print(file_name)

        if not os.path.isfile(file_name):
            num_runs = 1000
            all_params = []
            for i in range(num_runs):
                image1 = image.copy()
                image1.addNoise(galsim.GaussianNoise(sigma=noise))
                sd = piff.StarData(image1, image1.true_center, weight)
                s = piff.Star(sd, None)
                try:
                    s = model.initialize(s)
                    s = model.fit(s)
                except RuntimeError as e:  # Occasionally hsm fails.
                    print('Caught ',e)
                    continue
                print(s.fit.params)
                all_params.append(s.fit.params)
            var = np.var(all_params, axis=0)
            np.savez(file_name, var=var)
        var = np.load(file_name)['var']
        print('params = ',fit.params)
        print('empirical var = ',var)
        print('piff estimate = ',fit.params_var)
        print('ratio = ',fit.params_var/var)
        print('max ratio = ',np.max(fit.params_var/var))
        print('min ratio = ',np.min(fit.params_var/var))
        print('mean ratio = ',np.mean(fit.params_var/var))
        # Note: The fastfit=False estimates are better -- typically better than 10%
        #       The fastfit=True estimates are much rougher.  Especially size.  Need rtol=0.3.
        np.testing.assert_allclose(fit.params_var, var, rtol=0.3)

def test_fail():
    # Some vv noisy images that result in errors in the fit to check the error reporting.

    scale = 1.3
    g1 = 0.33
    g2 = -0.27
    flux = 15
    noise = 2.
    seed = 1234

    psf = galsim.Moffat(half_light_radius=1.0, beta=2.5, trunc=3.0)
    psf = psf.dilate(scale).shear(g1=g1, g2=g2).withFlux(flux)
    image = psf.drawImage(nx=64, ny=64, scale=0.3)

    weight = image.copy()
    weight.fill(1/noise**2)
    noisy_image = image.copy()
    rng = galsim.BaseDeviate(seed)
    noisy_image.addNoise(galsim.GaussianNoise(sigma=noise, rng=rng))

    star1 = piff.Star(piff.StarData(image, image.true_center, weight), None)
    star2 = piff.Star(piff.StarData(noisy_image, image.true_center, weight), None)

    model1 = piff.Moffat(fastfit=True, beta=2.5)
    with np.testing.assert_raises(RuntimeError):
        model1.initialize(star2)
    with np.testing.assert_raises(RuntimeError):
        model1.fit(star2)
    star3 = model1.initialize(star1)
    star3 = model1.fit(star3)
    star3 = piff.Star(star2.data, star3.fit)
    with np.testing.assert_raises(RuntimeError):
        model1.fit(star3)

    # This is contrived to hit the fit failure for the reference.
    # I'm not sure what realistic use case would actually hit it, but at least it's
    # theoretically possible to fail there.
    model2 = piff.GSObjectModel(galsim.InterpolatedImage(noisy_image), fastfit=True)
    with np.testing.assert_raises(RuntimeError):
        model2.initialize(star1)

    model3 = piff.Moffat(fastfit=False, beta=2.5, scipy_kwargs={'max_nfev':10})
    with np.testing.assert_raises(RuntimeError):
        model3.initialize(star2)
    with np.testing.assert_raises(RuntimeError):
        model3.fit(star2).fit
    star3 = model3.initialize(star1)
    star3 = model3.fit(star3)
    star3 = piff.Star(star2.data, star3.fit)
    with np.testing.assert_raises(RuntimeError):
        model3.fit(star3)

def test_gp():
    from test_gp_interp import test_gp_interp_isotropic
    test_gp_interp_isotropic()

@timer
def test_yaml():

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_gp.log')

    # Take DES test image, and test doing a psf run with GP interpolator
    # Use config parser:
    psf_file = os.path.join('output','gp_psf.fits')
    config = {
        'input' : {
            # These can be regular strings
            'image_file_name' : 'input/DECam_00241238_01.fits.fz',
            # Or any GalSim str value type.  e.g. FormattedStr
            'cat_file_name' : {
                'type': 'FormattedStr',
                'format': '%s/DECam_%08d_%02d_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits',
                'items': [
                    'input',    # dir
                    241238,     # expnum
                    1           # chipnum
                    ]
                },

            # What hdu is everything in?
            'image_hdu' : 1,
            'badpix_hdu' : 2,
            'weight_hdu' : 3,
            'cat_hdu' : 2,

            # What columns in the catalog have things we need?
            'x_col' : 'XWIN_IMAGE',
            'y_col' : 'YWIN_IMAGE',
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            'sky_col' : 'BACKGROUND',

            # How large should the postage stamp cutouts of the stars be?
            'stamp_size' : 21,
            },
        'psf' : {
            'model' : { 'type' : 'GSObjectModel',
                        'fastfit' : True,
                        'gsobj' : 'galsim.Gaussian(sigma=1.0)' },
            'interp' : { 'type' : 'GPInterp',
                         'keys' : ['u', 'v'],
                         'optimizer' : 'none',
                         'kernel' : 'RBF(200.0)'}
            },
        'output' : { 'file_name' : psf_file },
        }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)
    assert type(psf.model) == piff.GSObjectModel
    assert type(psf.interp) == piff.GPInterp
    print('nstars = ',len(psf.stars))
    target = psf.stars[17]
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, target.fit.params, decimal=3)
    # This should also work if the target doesn't have a fit yet.
    print('interpolate ',piff.Star(target.data,None))
    test_star = psf.interp.interpolate(piff.Star(target.data,None))
    np.testing.assert_almost_equal(test_star.fit.params, target.fit.params, decimal=3)


def get_correlation_length_matrix(correlation_length, g1, g2):
    """
    Produce correlation matrix to introduce anisotropy in kernel.
    Used same parametrization as shape measurement in weak-lensing
    because this is mathematicaly equivalent (anistropic kernel
    will have an elliptical shape).

    :param correlation_length: Correlation lenght of the kernel.
    :param g1, g2:             Shear applied to isotropic kernel.
    """
    if abs(g1)>1 or abs(g2)>1:
        raise ValueError('abs value of g1 and g2 must be lower than one')
    e = np.sqrt(g1**2 + g2**2)
    q = (1-e) / (1+e)
    m = galsim.Shear(g1=g2, g2=g2).getMatrix() * correlation_length
    L = m.dot(m) * q
    return L

def make_single_star(u, v, size, g1, g2, size_err, g1_err, g2_err):
    """Make a Star instance filled with a Kolmogorov profile

    :param u, v:           Star coordinate.
    :param size:           Star size.
    :param g1, g2:         Shear applied to profile.
    :param size_err:       Size error.
    :param g1_err, g2_err: Shear error.
    """
    star = piff.Star.makeTarget(x=None, y=None, u=u, v=v,
                                properties={}, wcs=None, scale=0.26,
                                stamp_size=24, image=None,
                                pointing=None, flux=1.)

    kolmogorov = galsim.Kolmogorov(half_light_radius=1., flux=1.)
    kolmogorov.drawImage(star.image, method='auto')
    fit = piff.StarFit(np.array([size, g1, g2]),
                       params_var=np.array([size_err**2, g1_err**2, g2_err**2]),
                       flux=1.)
    final_star = piff.Star(star.data, fit)

    return final_star

def return_gp_predict(y, X1, X2, kernel, factor):
    """Compute interpolation with gaussian process for a given kernel.

    :param y:      The dependent responses.  (n_samples, n_targets)
    :param X1:     The independent covariates.  (n_samples, 2)
    :param X2:     The independent covariates at which to interpolate.  (n_samples, 2)
    :param kernel: sklearn.gaussian_process kernel.
    :param factor: Cholesky decomposition of sklearn.gaussian_process kernel.
    """
    from scipy.linalg import cho_solve
    HT = kernel.__call__(X2, Y=X1)
    alpha = cho_solve(factor, y, overwrite_b=False)
    y_predict = np.dot(HT,alpha.reshape((len(alpha),1))).T[0]
    return y_predict

def make_gaussian_random_fields(kernel, nstars, noise_level=1e-3,
                                xlim=-10, ylim=10, seed=30352010,
                                test_size=0.20, vmax=8, plot=False):
    """
    Make psf params as gaussian random fields.

    :param kernel:      sklearn kernel to used for generating
                        the data.
    :param nstars:      number of stars to generate.
    :param noise_level: quantity of noise to add to the data.
    :param xlim:        x limit of the field.
    :param ylim:        y limit of the field.
    :param seed:        seed of the generator.
    :param test_size:   size ratio of the test sample.
    :param plot:        set to true to have plot of the field.
    :param vmax=8       max value for the color map.
    """
    from scipy.linalg import cholesky
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)

    # generate star coordinate
    if nstars<1500:
        nstars_interp = nstars
    else:
        nstars_interp = 1500
    u_interp = np.random.uniform(-xlim, xlim, nstars_interp)
    v_interp = np.random.uniform(-ylim, ylim, nstars_interp)
    coord_interp = np.array([u_interp, v_interp]).T

    # generate covariance matrix
    kernel = treegp.eval_kernel(kernel)
    cov_interp = kernel.__call__(coord_interp)

    # generate gaussian random fields
    size_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)
    g1_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)
    g2_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)

    if nstars<1500:
        size = size_interp
        g1 = g1_interp
        g2 = g2_interp
        u = u_interp
        v = v_interp
        coord = coord_interp
    else:
        # Interp on stars position using a gp interp with truth kernel.
        # Trick to have more stars faster as a gaussian random field.
        u = np.random.uniform(-xlim, xlim, nstars)
        v = np.random.uniform(-ylim, ylim, nstars)
        coord = np.array([u,v]).T

        K = kernel.__call__(coord_interp) + np.eye(nstars_interp)*1e-10
        factor = (cholesky(K, overwrite_a=True, lower=False), False)

        size = return_gp_predict(size_interp, coord_interp, coord, kernel, factor)
        g1 = return_gp_predict(g1_interp, coord_interp, coord, kernel, factor)
        g2 = return_gp_predict(g2_interp, coord_interp, coord, kernel, factor)

    # add noise on psfs parameters
    size += np.random.normal(scale=noise_level, size=nstars)
    g1 += np.random.normal(scale=noise_level, size=nstars)
    g2 += np.random.normal(scale=noise_level, size=nstars)

    size_err = np.ones(nstars)*noise_level
    g1_err = np.ones(nstars)*noise_level
    g2_err = np.ones(nstars)*noise_level

    # create stars
    stars = []
    for i in range(nstars):
        star = make_single_star(u[i], v[i],
                                size[i], g1[i], g2[i],
                                size_err[i], g1_err[i], g2_err[i])
        stars.append(star)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(u, v, c=size, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g1, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g2, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)

    # split training / validation
    stars_training, stars_validation = train_test_split(stars, test_size=test_size, random_state=42)

    return stars_training, stars_validation

def check_gp(stars_training, stars_validation, kernel, optimizer,
             min_sep=None, max_sep=None, nbins=20, l0=3000., rows=None,
             plotting=False, atol=4e-2, rtol=1e-3, test_star_fit=False):
    """ Solve for global PSF model, test it, and optionally display it.
    """
    print('start check_gp')
    interp = piff.GPInterp(kernel=kernel, optimizer=optimizer,
                           normalize=True, white_noise=0., l0=l0,
                           n_neighbors=4, average_fits=None, rows=rows,
                           nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                           logger=None)
    print('interp = ',interp)

    assert interp.property_names == ('u', 'v')

    interp.initialize(stars_training)
    print('after initialize')
    interp.solve(stars=stars_training, logger=None)
    print('after solve')

    if not test_star_fit:
        stars_test = interp.interpolateList(stars_validation)
    else:
        stars_v = copy.deepcopy(stars_validation)
        for s in stars_v:
            s.fit = None
        stars_test = interp.interpolateList(stars_v)
    print('made stars_test')

    xtest = np.array([interp.getProperties(star) for star in stars_validation])
    y_validation = np.array([star.fit.params for star in stars_validation])
    y_err = np.sqrt(np.array([star.fit.params_var for star in stars_validation]))

    y_test = np.array([star.fit.params for star in stars_test])

    np.testing.assert_allclose(y_test, y_validation, atol=atol)
    print('after some tests')

    if optimizer != 'none':
        truth_hyperparameters = np.exp(interp._init_theta)
        fitted_hyperparameters = np.exp(
                np.array([gp._optimizer._kernel.theta for gp in interp.gps]))
        np.testing.assert_allclose(np.mean(fitted_hyperparameters, axis=0),
                                   np.mean(truth_hyperparameters, axis=0),
                                   rtol=rtol)
    print('after optimizer')

    # Invalid kernel (can't use an instantiated kernel object for the kernel here)
    with np.testing.assert_raises(TypeError):
        piff.GPInterp(kernel=interp.gps[0].kernel, optimizer=optimizer)
    # Invalid optimizer
    with np.testing.assert_raises(ValueError):
        piff.GPInterp(kernel=kernel, optimizer='invalid')
    # Invalid number of kernels. (Can't tell until initialize)
    if isinstance(kernel, str):
        interp2 = piff.GPInterp(kernel=[kernel] * 4, optimizer=optimizer)
        with np.testing.assert_raises(ValueError):
            interp2.initialize(stars_training)
    print('after interp2')

    # Check I/O.
    file_name = os.path.join('output', 'test_gp.fits')
    with fitsio.FITS(file_name,'rw',clobber=True) as fout:
        interp.write(fout, extname='gp')
    with fitsio.FITS(file_name,'r') as fin:
        interp2 = piff.Interp.read(fin, extname='gp')

    stars_test = interp2.interpolateList(stars_validation)
    y_test = np.array([star.fit.params for star in stars_test])
    np.testing.assert_allclose(y_test, y_validation, atol=atol)
    print('after i/o')

    if plotting:
        import matplotlib.pyplot as plt
        title = ["size", "$g_1$", "$g_2$"]
        for j in range(3):
            plt.figure()
            plt.title('%s validation'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_validation[:,j], vmin=-4e-2, vmax=4e-2,
                        cmap=plt.cm.seismic)
            plt.colorbar()
            plt.figure()
            plt.title('%s test (gp interp)'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_test[:,j], vmin=-4e-2, vmax=4e-2,
                        cmap=plt.cm.seismic)
            plt.colorbar()

        if optimizer in ['isotropic', 'anisotropic']:
            if optimizer == 'isotropic':
                for gp in interp.gps:
                    plt.figure()
                    plt.scatter(gp._optimizer._2pcf_dist, gp._optimizer._2pcf)
                    plt.plot(gp._optimizer._2pcf_dist, gp._optimizer._2pcf_fit)
                    plt.plot(gp._optimizer._2pcf_dist,
                             np.ones_like(gp._optimizer._2pcf_dist)*4e-4,'b--')
                    plt.ylim(0,7e-4)
            else:
                for gp in interp.gps:
                    EXT = [np.min(gp._optimizer._2pcf_dist[:,0]),
                                  np.max(gp._optimizer._2pcf_dist[:,0]),
                           np.min(gp._optimizer._2pcf_dist[:,1]),
                                  np.max(gp._optimizer._2pcf_dist[:,1])]
                    CM = plt.cm.seismic
                    MAX = np.max(gp._optimizer._2pcf)
                    N = int(np.sqrt(len(gp._optimizer._2pcf)))

                    plt.figure(figsize=(10,5) ,frameon=False)
                    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)

                    plt.subplot(1,2,1)
                    plt.imshow(gp._optimizer._2pcf.reshape(N,N), extent=EXT,
                               interpolation='nearest', origin='lower',
                               vmin=-MAX, vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    plt.title('Measured 2-PCF',fontsize=16)

                    plt.subplot(1,2,2)
                    plt.imshow(gp._optimizer._2pcf_fit.reshape(N,N), extent=EXT,
                               interpolation='nearest',
                               origin='lower',vmin=-MAX,vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi\'$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)

        plt.show()
    print('after plotting')

@timer
def test_gp_interp_isotropic():

    print('start test_gp_interp_isotropic')
    if __name__ == "__main__":
        atol = 4e-2
        rtol = 3e-1
        nstars = [1600, 1600, 4000, 4000]
    else:
        atol = 4e-1
        rtol = 5e-1
        nstars = [160, 160, 400, 400]

    noise_level = 1e-3
    LIM = [10, 10, 20, 20]

    kernels = [["4e-4 * RBF(4.)", "4e-4 * RBF(4.)", "4e-4 * RBF(4.)"],
               "4e-4 * RBF(4.)",
               "4e-4 * RBF(4.)",
               "4e-4 * VonKarman(20.)"]

    optimizer = ['none',
                 'likelihood',
                 'isotropic',
                 'isotropic']

    rows = [[0,1,2], None, None, None]

    test_star_fit = [True, False, False, False]

    for i in range(len(kernels)):

        print('i = ',i)
        if i!=0:
            K = kernels[i]
        else:
            K = kernels[i][0]
        print('K = ',K)

        stars_training, stars_validation = make_gaussian_random_fields(
                K, nstars[i], xlim=-LIM[i], ylim=LIM[i],
                seed=30352010, vmax=4e-2, noise_level=noise_level)
        print('made stars')

        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], rows=rows[i],
                 atol=atol, rtol=rtol, test_star_fit=test_star_fit[i],
                 plotting=False)

@timer
def test_gp_interp_anisotropic():

    if __name__ == "__main__":
        atol = 4e-2
        rtol = 3e-1
        nstars = [1600, 4000, 1600, 4000]
    else:
        atol = 4e-1
        rtol = 5e-1
        nstars = [160, 500, 160, 500]

    noise_level = 1e-4

    L1 = get_correlation_length_matrix(4., 0.3, 0.3)
    invL1 = np.linalg.inv(L1)
    L2 = get_correlation_length_matrix(20., 0.3, 0.3)
    invL2 = np.linalg.inv(L2)

    kernels = ["4e-4 * AnisotropicRBF(invLam={0!r})".format(invL1),
               "4e-4 * AnisotropicRBF(invLam={0!r})".format(invL1),
               "4e-4 * AnisotropicVonKarman(invLam={0!r})".format(invL2),
               "4e-4 * AnisotropicVonKarman(invLam={0!r})".format(invL2)]

    optimizer = ['none',
                 'anisotropic',
                 'none',
                 'anisotropic']

    for i in range(len(kernels)):

        stars_training, stars_validation = make_gaussian_random_fields(
                kernels[i], nstars[i], xlim=-20, ylim=20,
                seed=30352010, vmax=4e-2,
                noise_level=noise_level)
        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], min_sep=0., max_sep=5., nbins=11,
                 l0=20., atol=atol, rtol=rtol, plotting=False)



if __name__ == '__main__':
    test_simple()
    test_center()
    test_interp()
    test_missing()
    test_gradient()
    test_gradient_center()
    test_direct()
    test_var()
    test_fail()
    #test_gp()
    test_yaml()
    test_gp_interp_anisotropic()
