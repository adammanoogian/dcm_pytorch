function [f,J] = spm_fx_cmc_nodelay(x,u,P,M)
% spm_fx_cmc_nodelay -- 2-output CMC state equation wrapper forcing D = I.
%
% [f,J] = spm_fx_cmc_nodelay(x,u,P,M)
%
% A thin pass-through to spm_fx_cmc that declares EXACTLY two output arguments.
% spm_fx_cmc.m:1 declares three outputs [f,J,D]; when an integrator probes
% nargout(M.f) >= 3 it pulls the delay operator D = spm_dcm_delay(P,M,J)
% (spm_fx_cmc.m:226), which -- even for a single source -- injects ~1 ms
% intrinsic inter-population delays (spm_dcm_delay.m:60-82, di = 1) so D ~= I.
%
% By exposing only 2 outputs here, spm_int_L.m:114 (`if nargout(f) >= 3`) is
% FALSE, so spm_int_L takes its :117 branch and keeps the delay operator at its
% initial value D = 1 (spm_int_L.m:112) -- an EXACT identity delay -- while still
% using spm_fx_cmc's own analytic Jacobian (spm_diff, spm_fx_cmc.m:206-208).
% This is preferred over stripping P.D (spm_dcm_delay.m:55,107,176), which leaves
% an exp(-16)-scale ~1e-7 round-off rather than exact D = I.
%
% Source: spm_int_L.m:112-122 (nargout/D branch), spm_dcm_delay.m:60-82 (the
% intrinsic-delay injection this wrapper avoids), spm_fx_cmc.m:1,206-226.

[f,J] = spm_fx_cmc(x,u,P,M);
