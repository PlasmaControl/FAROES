import openmdao.api as om
import numpy as np
from scipy.special import jv
from scipy.constants import pi
from openmdao.utils.assert_utils import assert_check_partials

Li1 = [-3.5115E+01, 1.9475E-01, 2.5082E-01, -1.6070E-01, 3.5190E-02]
bounds = [0.1, 100]
coeffs = [Li1]

N1 = [-3.4065E+01, -2.3614E+00, -6.0605E+00, -1.1570E+01, -6.9621E+00]
N2 = [-3.3899E+01, -5.9668E-01, 7.6272E-01, -1.7160E-01, 5.8770E-02]
N3 = [-3.3913E+01, -5.2628E-01, 7.0047E-01, -2.2790E-01, 2.8350E-02]
bounds = [0.1, 0.5, 2, 100]
coeffs = [N1, N2, N3]

slope_upper = sum(np.array([0, 1, 4, 12, 32]) * coeffs[-1])
extrap_upper = np.array([
    -2 * slope_upper + sum(np.array([1, 2, 4, 8, 16]) * coeffs[-1]),
    slope_upper, 0, 0, 0
])

slope_lower = sum(np.array([0, 1, -2, 3, -4]) * coeffs[0])
extrap_lower = np.array([
    slope_lower + sum(np.array([1, -1, 1, -1, 1]) * coeffs[0]), slope_lower, 0,
    0, 0
])


class RadiativeCoolingRate(om.ExplicitComponent):
    def setup(self):
        self.add_input("T", shape_by_conn=True, desc="temperature array")
        self.add_input("ne", shape_by_conn=True, desc="electron dens array")
        self.add_input("nZ", shape_by_conn=True, desc="ion dens array")
        self.add_input("dVdρ", shape_by_conn=True, desc="volume derivative")
        self.add_input("dρ", desc="rho differential")

        self.add_output("Lz", copy_shape="T", desc="rad cooling rate")
        self.add_output("P", desc="power")

    def compute(self, inputs, outputs):
        T_all = inputs["T"]
        ne_all = inputs["ne"]
        nZ_all = inputs["nZ"]
        dVdρ_all = inputs["dVdρ"]
        dρ = inputs["dρ"]

        X_all = np.log10(np.array(T_all))
        X_bounds = np.log10(np.array(bounds))

        new_X_bounds = np.insert(np.append(X_bounds, 10), 0, -10)

        new_coeffs1 = np.insert(np.array(coeffs), 0, extrap_lower, axis=0)
        new_coeffs = np.append(np.array(new_coeffs1), [extrap_upper], axis=0)

        X_vector_all = np.array(
            [np.ones_like(X_all), X_all, X_all**2, X_all**3, X_all**4])

        in_range = []
        for i in range(len(new_X_bounds) - 1):
            in_range.append(
                np.where(
                    np.logical_and(X_all > new_X_bounds[i],
                                   X_all < new_X_bounds[i + 1]), 1, 0))

        a1 = np.matmul(new_coeffs, X_vector_all)
        a2 = a1 * np.array(in_range)
        Lz = 10**sum(a2)
        #         outputs["Lz"] = Lz

        integrand = Lz * ne_all * nZ_all * dVdρ_all
        power = (2 * sum(integrand) - integrand[0] - integrand[-1]) * dρ / 2

        outputs["P"] = power

    def setup_partials(self):
        self.declare_partials("P", ["T"])

    def compute_partials(self, inputs, J):
        T_all = inputs["T"]
        ne_all = inputs["ne"]
        nZ_all = inputs["nZ"]
        dVdρ_all = inputs["dVdρ"]
        dρ = inputs["dρ"]

        X_all = np.log10(np.array(T_all))
        X_bounds = np.log10(np.array(bounds))

        new_X_bounds = np.insert(np.append(X_bounds, 10), 0, -10)

        new_coeffs1 = np.insert(np.array(coeffs), 0, extrap_lower, axis=0)
        new_coeffs = np.append(np.array(new_coeffs1), [extrap_upper], axis=0)

        X_vector_all = np.array(
            [np.ones_like(X_all), X_all, X_all**2, X_all**3, X_all**4])

        in_range = []
        for i in range(len(new_X_bounds) - 1):
            in_range.append(
                np.where(
                    np.logical_and(X_all > new_X_bounds[i],
                                   X_all < new_X_bounds[i + 1]), 1, 0))

        a1 = np.matmul(new_coeffs, X_vector_all)
        a2 = a1 * np.array(in_range)
        Lz = 10**sum(a2)

        X_derv_vector_all = np.array([
            np.zeros_like(X_all), 10**(-X_all), 2 * X_all * 10**(-X_all),
            3 * X_all**2 * 10**(-X_all), 4 * X_all**3 * 10**(-X_all)
        ])
        b1 = np.matmul(new_coeffs, X_derv_vector_all)
        b2 = b1 * np.array(in_range)
        chain_factor = sum(b2)
        integrand_dT = (Lz * chain_factor) * ne_all * nZ_all * dVdρ_all
        dPdt = (2 * sum(integrand_dT) - integrand_dT[0] -
                integrand_dT[-1]) * dρ / 2

        J["P", "T"] = dPdt


if __name__ == "__main__":

    ρlist = np.linspace(0, 1, 100)

    def parab_prof(ρ, init, α):
        return init * (1 - ρ**2)**α

    T0 = 200
    αT = 3
    T = np.delete(parab_prof(ρlist, T0, αT), -1)

    ne0 = 3e20
    αne = 4
    ne = np.delete(parab_prof(ρlist, ne0, αne), -1)

    nZ0 = 2e20
    αnZ = 5
    nZ = np.delete(parab_prof(ρlist, nZ0, αnZ), -1)

    R0 = 4
    a0 = 2.5
    δ0 = 0.3
    κ = 1.5

    def dVdρ(ρ):
        n1 = 8 * R0 * jv(0, δ0 * ρ) - 3 * a0 * ρ * jv(1, 2 * δ0 * ρ)
        n2 = 8 * R0 * jv(2, δ0 * ρ) - 3 * a0 * ρ * jv(3, 2 * δ0 * ρ)
        dVda = a0 * ρ * pi**2 * κ / 2 * (n1 + n2)

        n1 = a0 * ρ * jv(0, 2 * δ0 * ρ) + 2 * R0 * jv(1, δ0 * ρ)
        n2 = 2 * R0 * jv(3, δ0 * ρ) - a0 * ρ * jv(4, 2 * δ0 * ρ)
        dVdδ = (a0 * ρ)**2 * pi**2 * κ / 2 * (n1 + n2)

        dadρ = a0
        dδdρ = δ0

        return dVda * dadρ + dVdδ * dδdρ

    dVdρ = np.delete(dVdρ(ρlist), -1)

    prob = om.Problem()

    prob.model.add_subsystem("templist",
                             om.IndepVarComp("T", val=T),
                             promotes_outputs=["*"])
    prob.model.add_subsystem("nelist",
                             om.IndepVarComp("ne", val=ne),
                             promotes_outputs=["*"])
    prob.model.add_subsystem("nZlist",
                             om.IndepVarComp("nZ", val=nZ),
                             promotes_outputs=["*"])
    prob.model.add_subsystem("dVdrholist",
                             om.IndepVarComp("dVdρ", val=dVdρ),
                             promotes_outputs=["*"])

    prob.model.add_subsystem("radcooling",
                             RadiativeCoolingRate(),
                             promotes_inputs=["*"],
                             promotes_outputs=["*"])

    prob.setup(force_alloc_complex=True)

    prob.set_val("T", T)
    prob.set_val("ne", ne)
    prob.set_val("nZ", nZ)
    prob.set_val("dVdρ", dVdρ)
    prob.set_val("dρ", 0.01)

    check = prob.check_partials(method='cs')
    assert_check_partials(check)

    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
