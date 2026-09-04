"""The JAX RoadRunner model against the numba backend: flux, and autodiff against the analytic gradient."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import unittest

import numpy as np
import jax
import jax.numpy as jnp
from numpy import array, linspace, pi, concatenate, zeros, full, abs as npabs

from pytransit.models.rrmodel import RoadRunnerModel
from pytransit.backends.jax.rrmodel import rrmodel as jax_rrmodel
from pytransit.backends.numba.limb_darkening import ld_quadratic, ldi_quadratic, ldd_quadratic, ldig_quadratic

jax.config.update('jax_enable_x64', True)

P, A, INC = 3.0, 6.0, 1.45
K = array([0.10, 0.12])
LDC = array([[0.30, 0.10], [0.50, 0.15]])


def heterogeneous_data():
    """Two passbands with different supersampling, one epoch."""
    t = linspace(-0.09, 0.09, 150)
    times = concatenate([t, t])
    lcids = concatenate([zeros(t.size, int), full(t.size, 1, int)])
    return times, lcids, array([0, 1]), array([0, 0]), array([1, 4]), array([0.0, 0.01])


class TestJaxMatchesNumba(unittest.TestCase):
    def setUp(self):
        self.data = heterogeneous_data()
        self.jax = RoadRunnerModel(backend='jax')
        self.jax.set_data(*self.data[:4], nsamples=self.data[4], exptimes=self.data[5])
        self.numba = RoadRunnerModel(backend='numba', return_grad=True)
        self.numba.set_data(*self.data[:4], nsamples=self.data[4], exptimes=self.data[5])
        self.args = (K, 0.0, P, A, INC, 0.0, 0.0)
        # Expanded numba gradient columns: [k0 k1 | t0 | p a i e w | ld(pb0) ld(pb1)]
        self.cols = dict(k=(0, 1), t0=2, p=3, a=4, i=5, ld=8)
        self.pb_of_point = self.data[2][self.data[1]]

    def kernel(self, k, t0, p, a, i, ldp, ldi):
        times, lcids, pbids, epids, ns, et = self.data
        m = self.jax
        return jax_rrmodel(jnp.asarray(times), k, jnp.atleast_1d(t0), jnp.atleast_1d(p), jnp.atleast_1d(a), jnp.atleast_1d(i),
                           jnp.array([0.0]), jnp.array([0.0]), jnp.asarray(lcids), jnp.asarray(pbids), jnp.asarray(epids),
                           jnp.asarray(ns), jnp.asarray(et), ldp, None, ldi, None,
                           jnp.asarray(m._rules), m._pt0, m._pdt, m.ng, 2, 1, int(ns.max()))

    def ld_inputs(self):
        ldp = jnp.asarray(array([ld_quadratic(self.jax.mu, LDC[0]), ld_quadratic(self.jax.mu, LDC[1])]))
        ldi = jnp.asarray(array([ldi_quadratic(LDC[0]), ldi_quadratic(LDC[1])]))
        return ldp, ldi

    def test_flux_matches_the_numba_backend(self):
        flux_jax = np.asarray(self.jax.evaluate(*self.args, ldc=LDC))
        flux_numba = self.numba.evaluate(*self.args, ldc=LDC)[0]
        self.assertLess(flux_numba.min(), -1e-3)
        np.testing.assert_allclose(flux_jax, flux_numba, atol=1e-12)

    def test_autodiff_matches_the_analytic_gradient(self):
        flux, dflux = self.numba.evaluate(*self.args, ldc=LDC)
        ldp, ldi = self.ld_inputs()
        f = lambda k, t0, p, a, i: self.kernel(k, t0, p, a, i, ldp, ldi)
        jac = jax.jacfwd(f, argnums=(0, 1, 2, 3, 4))(jnp.asarray(K), 0.0, P, A, INC)
        jk = np.asarray(jac[0])[np.arange(flux.size), self.pb_of_point]
        dk = dflux[:, 0] + dflux[:, 1]
        # k includes the change of the mean intensity under the planet, which numba takes by a
        # forward difference; the orbital columns are exact on both sides (meepmeep's transit-centre
        # derivative is that of the Taylor polynomial, like the JAX autodiff).
        self._compare(jk, dk, rtol=1e-4, name='k')
        for name, j in zip(('t0', 'p', 'a', 'i'), jac[1:]):
            self._compare(np.asarray(j), dflux[:, self.cols[name]], rtol=1e-6, name=name)

    def test_autodiff_matches_the_limb_darkening_columns(self):
        flux, dflux = self.numba.evaluate(*self.args, ldc=LDC)
        ldp, ldi = self.ld_inputs()
        f = lambda ldp, ldi: self.kernel(jnp.asarray(K), 0.0, P, A, INC, ldp, ldi)
        j_ldp, j_ldi = jax.jacfwd(f, argnums=(0, 1))(ldp, ldi)
        for ipb in range(2):
            ldg = ldd_quadratic(self.jax.mu, LDC[ipb])[1:]   # (nldc, nmu)
            dldi = ldig_quadratic(LDC[ipb])
            for j in range(2):
                chain = np.asarray(j_ldp)[:, ipb] @ ldg[j] + np.asarray(j_ldi)[:, ipb] * dldi[j]
                self._compare(chain, dflux[:, self.cols['ld'] + ipb * 2 + j], rtol=1e-6, name=f'ld pb{ipb} c{j}')

    def test_all_derivatives_are_finite(self):
        ldp, ldi = self.ld_inputs()
        f = lambda k, t0, p, a, i: self.kernel(k, t0, p, a, i, ldp, ldi)
        for arr in jax.jacfwd(f, argnums=(0, 1, 2, 3, 4))(jnp.asarray(K), 0.0, P, A, INC):
            self.assertTrue(np.isfinite(np.asarray(arr)).all())

    def _compare(self, got, expected, rtol, name):
        m = npabs(expected) > 1e-6 * npabs(expected).max()
        self.assertTrue(m.any(), name)
        np.testing.assert_allclose(got[m], expected[m], rtol=rtol, atol=1e-10, err_msg=name)


class TestPopulationEvaluation(unittest.TestCase):
    def test_batched_evaluation_matches_single(self):
        times, lcids, pbids, epids, ns, et = heterogeneous_data()
        model = RoadRunnerModel(backend='jax')
        model.set_data(times, lcids, pbids, epids, nsamples=ns, exptimes=et)
        ks = array([[0.08, 0.09], [0.10, 0.12], [0.20, 0.20]])
        incs = array([0.5 * pi, 1.45, 1.42])
        batch = np.asarray(model.evaluate(ks, zeros(3), full(3, P), full(3, A), incs, zeros(3), zeros(3), ldc=LDC))
        self.assertEqual(batch.shape, (3, times.size))
        for ipv in range(3):
            single = np.asarray(model.evaluate(ks[ipv], 0.0, P, A, incs[ipv], 0.0, 0.0, ldc=LDC))
            np.testing.assert_allclose(batch[ipv], single, atol=1e-14)

    def test_more_nodes_change_the_flux_little(self):
        times = linspace(-0.09, 0.09, 100)
        coarse = RoadRunnerModel(backend='jax', nq=8, ng=100)
        fine = RoadRunnerModel(backend='jax', nq=16, ng=400)
        for m in (coarse, fine):
            m.set_data(times)
        fc = np.asarray(coarse.evaluate(0.2, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC[0]))
        ff = np.asarray(fine.evaluate(0.2, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC[0]))
        self.assertLess(npabs(fc - ff).max(), 5e-6)


if __name__ == '__main__':
    unittest.main()
