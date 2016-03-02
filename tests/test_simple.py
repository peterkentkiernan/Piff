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
import numpy
import piff
import os
import fitsio

def test_Gaussian():
    """This is about the simplest possible model I could think of.  It just uses the
    HSM adaptive moments routine to measure the moments, and then it models the
    PSF as a Gaussian.
    """

    # Here is the true PSF
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)
    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')

    # Make a StarData instance for this image
    star = piff.StarData(image, image.center())

    # Fit the model from the image
    model = piff.Gaussian()
    model.fitStar(star)

    print('True sigma = ',sigma,', model sigma = ',model.sigma)
    print('True g1 = ',g1,', model g1 = ',model.shape.g1)
    print('True g2 = ',g2,', model g2 = ',model.shape.g2)

    # This test is pretty accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Gaussian.
    true_params = [ sigma, g1, g2 ]
    numpy.testing.assert_almost_equal(model.sigma, sigma, decimal=7)
    numpy.testing.assert_almost_equal(model.shape.g1, g1, decimal=7)
    numpy.testing.assert_almost_equal(model.shape.g2, g2, decimal=7)
    numpy.testing.assert_almost_equal(model.getParameters(), true_params, decimal=7)

    # Now test running it via the config parser
    config = {
        'model' : {
            'type' : 'Gaussian'
        }
    }
    logger = piff.config.setup_logger()
    model = piff.process_model(config, logger)
    model.fitStar(star)

    # Same tests.
    numpy.testing.assert_almost_equal(model.sigma, sigma, decimal=7)
    numpy.testing.assert_almost_equal(model.shape.g1, g1, decimal=7)
    numpy.testing.assert_almost_equal(model.shape.g2, g2, decimal=7)
    numpy.testing.assert_almost_equal(model.getParameters(), true_params, decimal=7)


def test_Mean():
    """For the interpolation, the simplest possible model is just a mean value, which barely
    even qualifies as doing any kind of interpolating.  But it tests the basic glue software.
    """
    import numpy
    # Make a list of paramter vectors to "interpolate"
    numpy.random.seed(123)
    nstars = 100
    vectors = [ numpy.random.random(10) for i in range(nstars) ]
    mean = numpy.mean(vectors, axis=0)

    # Give each parameter vector a position
    pos = [ galsim.PositionD(numpy.random.random()*2048, numpy.random.random()*2048)
            for i in range(nstars) ]

    # Use the piff.Mean interpolator
    interp = piff.Mean()
    interp.solve(pos, vectors)

    print('True mean = ',mean)
    print('Interp mean = ',interp.mean)

    # This should be exactly equal, since we did the same calculation.  But use almost_equal
    # anyway, just in case we decide to do something slightly different, but equivalent.
    numpy.testing.assert_almost_equal(mean, interp.mean)

    # Now test running it via the config parser
    config = {
        'interp' : {
            'type' : 'Mean'
        }
    }
    logger = piff.config.setup_logger()
    interp = piff.process_interp(config, logger)
    interp.solve(pos, vectors)
    numpy.testing.assert_almost_equal(mean, interp.mean)


def test_single_image():
    """Test the simple case of one image and one catalog.
    """
    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars.  Include some flagged and not used locations.
    x_list = [ 123, 345, 567, 1094, 924, 1532, 1743, 888, 1033, 1409 ]
    y_list = [ 345, 567, 1094, 924, 1532, 1743, 888, 1033, 1409, 123 ]
    flag_list = [ 0, 0, 12, 0, 0, 1, 0, 0, 0, 0 ]
    use_list = [ 1, 1, 1, 1, 1, 0, 1, 1, 0, 1 ]

    # Draw a Gaussian PSF at each location on the image.
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    for x,y,flag,use in zip(x_list, y_list, flag_list, use_list):
        bounds = galsim.BoundsI(x-32, x+32, y-32, y+32)
        psf.drawImage(image=image[bounds], method='no_pixel')
        # corrupt the ones that are marked as flagged
        if flag:
            print('corrupting star at ',x,y)
            ar = image[bounds].array
            im_max = numpy.max(ar) * 0.2
            ar[ar > im_max] = im_max

    # Write out the image to a file
    image_file = os.path.join('data','simple_image.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8'), ('flag','i2'), ('use','i2') ]
    data = numpy.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    data['flag'] = flag_list
    data['use'] = use_list
    cat_file = os.path.join('data','simple_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # Use InputFiles to read these back in
    input = piff.InputFiles(image_file, cat_file)
    assert input.image_files == [ image_file ]
    assert input.cat_files == [ cat_file ]
    assert input.x_col == 'x'
    assert input.y_col == 'y'

    # Check image
    input.readImages()
    assert len(input.images) == 1
    numpy.testing.assert_equal(input.images[0].array, image.array)

    # Check catalog
    input.readStarCatalogs()
    assert len(input.cats) == 1
    numpy.testing.assert_equal(input.cats[0]['x'], x_list)
    numpy.testing.assert_equal(input.cats[0]['y'], y_list)

    # Repeat, using flag and use columns this time.
    input = piff.InputFiles(image_file, cat_file, flag_col='flag', use_col='use', stamp_size=48)
    assert input.flag_col == 'flag'
    assert input.use_col == 'use'
    input.readImages()
    input.readStarCatalogs()
    assert len(input.cats[0]) == 7

    # Make star data
    stars = input.makeStarData()
    assert len(stars) == 7
    assert stars[0].image.array.shape == (48,48)

    # Process the star data
    model = piff.Gaussian()
    interp = piff.Mean()
    vectors = [ model.fitStar(star).getParameters() for star in stars ]
    print('vectors = ',vectors)
    pos = [ interp.getStarPosition(star) for star in stars ]
    print('pos = ',pos)
    interp.solve(pos, vectors)
    print('mean = ',interp.mean)

    # Check that the interpolation is what it should be
    target = numpy.array([ 10, 10 ])  # Any position would work here.
    true_params = [ sigma, g1, g2 ]
    test_params = interp.interpolate(target)
    numpy.testing.assert_almost_equal(test_params, true_params, decimal=5)

    # Now test running it via the config parser
    config = {
        'input' : {
            'images' : image_file,
            'cats' : cat_file,
            'flag_col' : 'flag',
            'use_col' : 'use',
            'stamp_size' : 48
        }
    }
    logger = piff.config.setup_logger()
    stars = piff.process_input(config, logger)

    vectors = [ model.fitStar(star).getParameters() for star in stars ]
    pos = [ interp.getStarPosition(star) for star in stars ]
    interp.solve(pos, vectors)
    test_params = interp.interpolate(target)
    numpy.testing.assert_almost_equal(test_params, true_params, decimal=5)


if __name__ == '__main__':
    test_Gaussian()
    test_Mean()
    test_single_image()

