"""
Python translation of methodAFit C++ code.
This module provides a simplified implementation of the algorithm
that was originally implemented in C++ using ROOT.  Histograms are
represented by numpy arrays and several physics helper functions
mirror the original C++ functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Physical constants (values taken from the C++ code)
class physics:
    me = 510998.95  # electron mass in eV
    mn = 939565413.3  # neutron mass in eV
    delta = 1293.332e3  # neutron-proton mass difference in eV
    alpha_SI = 1/137.035999139
    pi = math.pi
    t2factor = 1e12  # used for TOF^2 -> momentum conversion

    @staticmethod
    def pe(Ee: float) -> float:
        return math.sqrt((Ee + physics.me) ** 2 - physics.me ** 2)

    @staticmethod
    def pe2(Ee: float) -> float:
        return (Ee + physics.me) ** 2 - physics.me ** 2

    @staticmethod
    def pv(Ee: float) -> float:
        return physics.delta - physics.me - Ee

    @staticmethod
    def ppmin(Ee: float) -> float:
        return physics.pe(Ee) - physics.pv(Ee)

    @staticmethod
    def ppmax(Ee: float) -> float:
        return physics.pe(Ee) + physics.pv(Ee)

    @staticmethod
    def pp2diff(Ee: float) -> float:
        return 4 * physics.pe(Ee) * physics.pv(Ee)

    @staticmethod
    def ppmid(Ee: float) -> float:
        return math.sqrt(physics.pe2(Ee) + physics.pv(Ee)**2)

    @staticmethod
    def beta(Ee: float) -> float:
        if Ee > 0:
            return physics.pe(Ee) / (physics.me + Ee)
        return 0.0

    @staticmethod
    def gamma_C(Ee: float) -> float:
        b = physics.beta(Ee)
        return 1 / math.sqrt(1 - b*b)


# Helper functions

def eEspec(x: float) -> float:
    """Beta decay electron spectrum (simplified)."""
    if 0 < x < physics.delta - physics.me:
        beta_val = physics.beta(x)
        corr = (2 * physics.pi * physics.alpha_SI / beta_val)
        corr /= 1 - math.exp(-2 * physics.pi * physics.alpha_SI / beta_val)
        return (physics.delta - physics.me - x) ** 2 * physics.pe(x) * (x + physics.me) * corr
    return 0.0


def eECal(e2: float, offset: float, gain: float, nonlin: float) -> float:
    energy_adc = offset + gain * e2 + (nonlin / 1e6) * gain * gain * e2 * e2
    if energy_adc < 0 or energy_adc > physics.delta - physics.me:
        return 0.0
    return energy_adc


def ensure_range(val: float, min_v: float, max_v: float) -> float:
    if val < min_v:
        return min_v
    if val > max_v:
        return max_v
    return val


@dataclass
class MethodAFit:
    e2_start: float = 0.0
    e2_step: float = 1.0
    e2_npts: int = 10
    Et2_start: float = 0.0
    Et2_step: float = 1.0
    Et2_npts: int = 10
    cosThetaMax: float = 1.0
    npos: int = 1

    chan: np.ndarray = field(init=False)
    eESpec_raw: List[float] = field(init=False)
    eESpec_me: List[float] = field(init=False)
    lenSpec: int = 100
    tailStruct: List[float] = field(default_factory=lambda: [0.01, 7e-5, 0.13])
    hvMapping: List[float] = field(default_factory=lambda: [0.0]*6)
    ybCorr: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def __post_init__(self):
        self.chan = np.zeros(self.e2_npts * self.Et2_npts)
        self.eESpec_raw = [0.0 for _ in range(self.lenSpec+1)]
        self.eESpec_me = [0.0 for _ in range(self.lenSpec+1)]
        temp1 = 0.0
        temp2 = 0.0
        availE = physics.delta - physics.me
        for ie in range(1, self.lenSpec+1):
            mid_val = (ie-0.5)/self.lenSpec * availE
            spec_val = eEspec(mid_val)
            temp1 += spec_val
            self.eESpec_raw[ie] = temp1
            temp2 += spec_val * physics.me / (physics.me + mid_val)
            self.eESpec_me[ie] = temp2
        spec_norm = self.eESpec_raw[self.lenSpec]
        for ie in range(self.lenSpec+1):
            self.eESpec_raw[ie] /= spec_norm
            self.eESpec_me[ie] /= spec_norm

    # ---- utility conversion functions ----
    def Et2_to_di(self, Et2: float) -> float:
        return (Et2 - self.Et2_start) / self.Et2_step

    def di_to_Et2(self, i: float) -> float:
        return i * self.Et2_step + self.Et2_start

    def e2_to_di(self, e2: float) -> float:
        return (e2 - self.e2_start) / self.e2_step

    def di_to_e2(self, i: float) -> float:
        return i * self.e2_step + self.e2_start

    # ---- physics related calculations ----
    def TOF_MethodA(self, cosTh: float, pp2: float, par: List[float]) -> float:
        A_cosThetaMin, L_zDV, A_alpha, A_beta, A_gamma, A_eta = par
        cosmid = (1 + A_cosThetaMin) / 2
        if cosTh <= A_cosThetaMin:
            cosTh += 1e-6
        result = (
            L_zDV
            - A_eta * math.log((cosTh - A_cosThetaMin) / (1 - A_cosThetaMin))
            - A_alpha * (cosTh - cosmid)
            + A_beta * (cosTh - cosmid) ** 2
            - A_gamma * (cosTh - cosmid) ** 3
        )
        if pp2 != 0:
            hv_corr = self.HVCorrection(pp2 / 1e12)
            result *= hv_corr
        return result

    def HVCorrection(self, pp2: float) -> float:
        coeffs = self.hvMapping
        return (
            1
            + 1e-2 * coeffs[0] / pp2
            + 1e-3 * coeffs[1]
            + 1e-3 * coeffs[2] * pp2
            + 1e-6 * coeffs[3] * pp2**2
            + 1e-8 * coeffs[4] * pp2**3
            + 1e-11 * coeffs[5] * pp2**4
        )

    # simplified FillChannels using numpy arrays
    def FillChannels(self, i0: int, cos_min: float, cos_max: float, pp2: float, intens: float, paramsA: List[float]):
        tp_low = self.TOF_MethodA(cos_min, pp2, paramsA)
        tp_high = self.TOF_MethodA(cos_max, pp2, paramsA)
        Et2_low = physics.t2factor * pp2 / (tp_high * tp_high)
        Et2_high = physics.t2factor * pp2 / (tp_low * tp_low)
        lBnd = self.Et2_to_di(Et2_low)
        rBnd = self.Et2_to_di(Et2_high)
        il = int(math.floor(lBnd))
        ir = int(math.floor(rBnd))
        if il == ir:
            if 0 <= il < self.Et2_npts:
                self.chan[i0 + il] += intens * (rBnd - lBnd) * (cos_max - cos_min)
        else:
            for it2 in range(max(il, 0), min(ir + 1, self.Et2_npts)):
                self.chan[i0 + it2] += intens * (cos_max - cos_min)

    def getSpecFierz(self, bin: int, b: float) -> float:
        if 0 <= bin <= self.lenSpec:
            return self.eESpec_raw[bin] + b * self.eESpec_me[bin]
        return 0.0

    def numElectrons(self, eMin: float, eMax: float, b: float) -> float:
        availE = physics.delta - physics.me
        eStep = availE / self.lenSpec
        if eMin > availE:
            return 0.0
        binL = int(ensure_range(math.floor(eMin / eStep - 0.5), 0, self.lenSpec))
        binR = int(ensure_range(math.floor(eMax / eStep - 0.5), 0, self.lenSpec))
        resultL = ((eMin / eStep) - binL) * (
            self.getSpecFierz(binL + 1, b) - self.getSpecFierz(binL, b)
        ) + self.getSpecFierz(binL, b)
        resultR = ((eMax / eStep) - binR) * (
            self.getSpecFierz(binR + 1, b) - self.getSpecFierz(binR, b)
        ) + self.getSpecFierz(binR, b)
        if resultR > resultL:
            return (resultR - resultL) / self.getSpecFierz(self.lenSpec, b)
        return 0.0

    def simulateET2SpecMethodA(self, params: List[float]):
        a_ev, b_F, log_intens = params[:3]
        intens = 10 ** log_intens
        paramsA = [params[3], params[4] + 5.0, params[5], params[6], params[7], params[8]]
        z0_center = params[9]
        z0_width = params[10]
        self.tailStruct[0] = params[11]
        self.tailStruct[1] = params[12] * 1e-4
        self.tailStruct[2] = params[13]
        self.hvMapping = params[14:20]
        calEe = params[20]
        EeNonLinearity = params[21]
        for ie in range(self.e2_npts):
            ioffset = ie * self.Et2_npts
            eE = eECal(ie + 0.5, self.e2_start, calEe * self.e2_step, EeNonLinearity)
            pp2max = physics.ppmax(eE) ** 2
            pp2min = physics.ppmin(eE) ** 2
            for ipos in range(self.npos):
                z_tmp = z0_center + z0_width * ((ipos + 0.5) / self.npos - 0.5)
                paramsA[1] = params[4] + 5.0 - z_tmp
                for pp2 in np.linspace(pp2min, pp2max, 5):
                    P_p2 = 1 + a_ev * (pp2 - (pp2max + pp2min) / 2) / (
                        2 * (eE + physics.me) * (physics.delta - physics.me - eE)
                    )
                    intens_tmp = intens * P_p2 / (self.npos * 5) / 2
                    self.FillChannels(ioffset, paramsA[0] + 1e-6, self.cosThetaMax, pp2, intens_tmp, paramsA)



