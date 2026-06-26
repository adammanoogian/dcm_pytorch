function [f,J] = spm_fx_cmc_nodelay(x,u,P,M)
% spm_fx_cmc_nodelay -- 2-output CMC state equation wrapper forcing D = I.
%
% [f,J] = spm_fx_cmc_nodelay(x,u,P,M)
%
% A nargout-aware pass-through to spm_fx_cmc that DECLARES exactly two output
% arguments. When an integrator probes nargout(M.f) >= 3 it pulls the delay
% operator D = spm_dcm_delay(P,M,J) (spm_fx_cmc.m:226), which -- even for a single
% source -- injects ~1 ms intrinsic inter-population delays (spm_dcm_delay.m:60-82,
% di = 1) so D ~= I. By exposing only 2 declared outputs here, spm_int_L.m:114
% (`if nargout(f) >= 3`) is FALSE, so spm_int_L takes its :117 branch and keeps the
% delay operator at its initial value D = 1 (spm_int_L.m:112) -- an EXACT identity
% delay -- while still using spm_fx_cmc's own analytic Jacobian.
%
% The nargout guard is LOAD-BEARING (NOT cosmetic). spm_fx_cmc computes its
% analytic Jacobian via spm_diff(M.f,x,u,P,M,1) (spm_fx_cmc.m:208) -- it
% differentiates M.f, which here points back at THIS wrapper. spm_diff probes the
% target with a single output (nargout == 1) precisely so spm_fx_cmc's Jacobian
% block (`if nargout < 2, return`) short-circuits and the recursion terminates. An
% unconditional `[f,J] = spm_fx_cmc(...)` body would force the 2-output path on
% EVERY call regardless of the wrapper's own nargout, so spm_fx_cmc would always
% re-enter spm_diff(M.f) -> wrapper -> spm_fx_cmc(2 outputs) -> ... ad infinitum
% (observed: "Out of memory / infinite recursion", stack spm_fx_cmc_nodelay:22 ->
% spm_diff:92 -> spm_fx_cmc:208 -> ...). Guarding on nargout preserves the standard
% SPM single-output termination while keeping nargout(M.f) == 2 for the D = I route.
%
% Source: spm_int_L.m:112-122 (nargout/D branch), spm_fx_cmc.m:206-226 (the analytic
% Jacobian via spm_diff(M.f,...)), spm_dcm_delay.m:60-82 (the delay injection avoided).

if nargout < 2
    f = spm_fx_cmc(x,u,P,M);
else
    [f,J] = spm_fx_cmc(x,u,P,M);
end
