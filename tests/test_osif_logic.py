import unittest

import numpy as np

from osif.logic import EisLogic


class EisLogicFaradaicTmlTests(unittest.TestCase):
    def setUp(self):
        self.logic = EisLogic()

    def test_nonfaradaic_transmission_line_matches_legacy_equation(self):
        freq = np.geomspace(0.1, 1e4, 30)
        params = [0.0, 0.17, 0.42, 0.08, 0.86, 0.0, np.inf]

        z_model = self.logic.evaluate_model("Transmission Line", params, freq)

        omega = 1j * 2 * np.pi * freq
        x = np.sqrt(params[2] * params[3] * (omega ** params[4]))
        prefactor = np.sqrt(params[2] / (params[3] * (omega ** params[4])))
        z_expected = params[1] + prefactor * self.logic.JPcoth(x)

        np.testing.assert_allclose(z_model, z_expected, rtol=1e-12, atol=1e-12)

    def test_faradaic_large_rk_converges_to_nonfaradaic_tml(self):
        freq = np.geomspace(0.1, 1e4, 30)
        nf_params = [0.0, 0.12, 0.35, 0.06, 0.88, 0.0, np.inf]
        f_params = [0.0, 0.12, 0.35, 0.06, 0.88, 0.0, 1e30]

        z_nf = self.logic.evaluate_model("Transmission Line", nf_params, freq)
        z_f = self.logic.evaluate_model(self.logic.FARADAIC_TML_MODEL, f_params, freq)

        np.testing.assert_allclose(z_f, z_nf, rtol=1e-10, atol=1e-10)

    def test_faradaic_phi_one_matches_ideal_capacitance_equation(self):
        freq = np.geomspace(1.0, 1e4, 25)
        lwire, hfr, rcl, cdl, phi, theta, rk = [0.0, 0.08, 0.31, 0.04, 1.0, 0.0, 2.5]
        params = [lwire, hfr, rcl, cdl, phi, theta, rk]

        z_model = self.logic.evaluate_model(self.logic.FARADAIC_TML_MODEL, params, freq)

        omega = 1j * 2 * np.pi * freq
        zk = rk / (1.0 + omega * rk * cdl)
        z_expected = hfr + np.sqrt(rcl * zk) * self.logic.JPcoth(np.sqrt(rcl / zk))

        np.testing.assert_allclose(z_model, z_expected, rtol=1e-12, atol=1e-12)

    def test_nonfaradaic_low_frequency_intercept_is_rcl_over_three(self):
        freq = np.geomspace(1e-7, 1e-5, 8)
        hfr, rcl, qdl, phi = [0.0, 0.9, 0.2, 0.82]
        params = [0.0, hfr, rcl, qdl, phi, 0.0, np.inf]

        z_model = self.logic.evaluate_model("Transmission Line", params, freq)
        omega = 1j * 2 * np.pi * freq
        capacitive_term = 1.0 / (qdl * (omega ** phi))
        intercept = np.real(z_model - hfr - capacitive_term)

        np.testing.assert_allclose(intercept, rcl / 3.0, rtol=1e-4, atol=1e-6)

    def test_fit_impedance_accepts_faradaic_vector_and_returns_rk(self):
        freq = np.geomspace(1.0, 1e3, 12)
        init = [0.1, 0.4, 0.05, 0.9, 1.8]
        canonical = [0.0, init[0], init[1], init[2], init[3], 0.0, init[4]]
        z_exp = self.logic.evaluate_model(self.logic.FARADAIC_TML_MODEL, canonical, freq)

        results, error = self.logic.fit_impedance(
            self.logic.FARADAIC_TML_MODEL,
            init,
            freq,
            z_exp,
            max_nfev=20,
            n_restarts=1,
            fit_inductance=False,
        )

        self.assertIsNone(error)
        params, se, residuals, info = results
        self.assertEqual(len(params), 7)
        self.assertEqual(len(se), 7)
        self.assertAlmostEqual(params[6], init[4], places=9)
        self.assertTrue(info["fit_faradaic"])
        self.assertEqual(info["ranking_basis"], "hfr_rcl_se")
        self.assertLess(float(residuals @ residuals), 1e-20)


if __name__ == "__main__":
    unittest.main()
