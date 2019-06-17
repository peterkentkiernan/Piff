import piff
import galsim
import numpy as np
from piff_test_helper import timer


@timer
def do(star):
    orig_star = star.copy()
    for _ in range(2000):
        h2 = piff.util.hsm(star)
        dh2 = piff.util.hsm_error(star)
        h3 = piff.util.hsm_third_moments(star)
        dh3 = piff.util.hsm_error_third_moments(star)
        h4 = piff.util.hsm_fourth_moments(star)
        dh4 = piff.util.hsm_error_fourth_moments(star)
        ho = piff.util.hsm_orthogonal(star)
        dho = piff.util.hsm_error_orthogonal(star)
        assert star.data.image == orig_star.data.image
    # regression test against commit 877310b
    np.testing.assert_allclose(
        h2,
        (52092.76953125, 0.3238881590779439, 0.37834976781419183, 0.45353370987659497,
         -0.0034246192313707063, 0.013793631060302557, 0)
    )
    np.testing.assert_allclose(
        dh2,
        (39330.846230404364, 0.3238881429661985, 0.3783496878718334, 0.41155247472496903,
         -0.0028181793848802705, 0.011351564441240823, 423.5136428965855, 0.006052274365939154,
         0.006779770176601579, 0.0058591803135446445, 0.005765225406908892, 0.005765806767807908)
    )
    np.testing.assert_allclose(
        h3,
        (39330.846230404364, 0.3238881429661985, 0.3783496878718334, 0.41155247472496903,
         -0.0028181793848802705, 0.011351564441240823, 0.003677924054913353, 0.0016070498688819789,
         0.005499451348497089, 0.004523034833675351)
    )
    np.testing.assert_allclose(
        dh3,
        (423.5136428965855, 0.006052274365939154, 0.006779770176601579, 0.0058591803135446445,
         0.005765225406908892, 0.005765806767807908, 0.0009113473040058723, 0.0009915664874654774,
         0.0018868586018156953, 0.0018861795391844946)
    )
    np.testing.assert_allclose(
        h4,
        (39330.846230404364, 0.3238881429661985, 0.3783496878718334, 0.41155247472496903,
         -0.0028181793848802705, 0.011351564441240823, 0.003677924054913353, 0.0016070498688819789,
         0.005499451348497089, 0.004523034833675351, 0.17878917848015433, 0.0009076435116469328,
         0.008656040263549082, -0.0007599979020067316, 0.000989424831262702)
    )
    np.testing.assert_allclose(
        dh4,
        (423.5136428965855, 0.006052274365939154, 0.006779770176601579, 0.0058591803135446445,
         0.005765225406908892, 0.005765806767807908, 0.0009113473040058723, 0.0009915664874654774,
         0.0018868586018156953, 0.0018861795391844946, 0.005784901600889518, 0.004107559302187891,
         0.004121890464473664, 0.0018038101104597684, 0.0018037940984628332)
    )
    np.testing.assert_allclose(
        ho,
        (39330.846230404364, 0.3238881429661985, 0.3783496878718334, 0.41155247472496903,
         -0.0028181793848802705, 0.011351564441240823, 0.003677924054913353, 0.0016070498688819789,
         0.005499451348497089, 0.004523034833675351, -1.0558682456947528, 3.6323692616162035,
         -15.702952564846111)
    )
    np.testing.assert_allclose(
        dho,
        (423.5136428965855, 0.006052274365939154, 0.006779770176601579, 0.0058591803135446445,
         0.005765225406908892, 0.005765806767807908, 0.0009113473040058723, 0.0009915664874654774,
         0.0018868586018156953, 0.0018861795391844946, 0.0079292469510418, 0.011214822899852369,
         0.0718401321790962)
    )


if __name__ == '__main__':
    psf = galsim.Convolve(
        galsim.OpticalPSF(lam=750.0, diam=4.2, obscuration=0.3, aberrations=[0]+[0.1]*22),
        galsim.Kolmogorov(lam=750.0, r0_500=0.1)
    ).shift(0.02, 0.04)
    wcs = galsim.JacobianWCS(*np.array([0.98, 0.02, 0.02, 0.98])*0.2)
    img = psf.drawImage(nx=32, ny=32, wcs=wcs)
    rng = galsim.BaseDeviate(577)
    var = img.addNoiseSNR(galsim.CCDNoise(rng, sky_level=1000, read_noise=5.0), 200)
    weight = galsim.Image(np.ones_like(img.array)/var)

    star = piff.Star(
        piff.StarData(img, galsim.PositionD(15, 15), weight=weight),
        piff.StarFit(None)
    )

    import cProfile
    import pstats
    import subprocess
    pr = cProfile.Profile()
    pr.enable()

    do(star)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(30)
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(30)

    pr.dump_stats("hsm.prof")
    cmd = "gprof2dot -f pstats hsm.prof -n1 -e1 | dot -Tpng -o hsm.png"
    subprocess.run(cmd, shell=True)
