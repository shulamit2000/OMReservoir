function dydt = fn_coupledsystem_diff_equations(t,y,Pin,lambdaL, E_inj,E_tau, kappa_fb, kappa_inj,DTcoeff)
c=299792458;
hbar=1.054e-34;

U_re=y(1);
U_im=y(2);
N = y(3);
DT = y(4)*DTcoeff;
X = y(5);
Xprime = y(6);

U=U_re+1i*U_im;

% Fixed parameters
dnsi_dT=1.86e-4; %1.86e-4;
dnsi_dN=-1.73e-27;
nSi=3.476;
beta_si=8.4e-12;
sigma_si=1e-21;
rho_si=2.33e3;
Cp=700;

lambda0=  1572.8e-9; %1584.15e-9  ; % 
gamma_i=19e9; %40E9 ; %19e9; %1.58E9 ; 
gamma_e=2*pi*2.2E09; %9.28E09; %5e9 2.2E09;
Omega_m=2*pi*110e6 ; %6.8e9; %*110e6; 2*pi*110e6 ;%
g0= 690e3; %1e6; %533e3; %1e6; %
Gamma_TPA=0.8012;
V_TPA=6.4e-19;
m_eff=2.4e-14; %3.0272E-16; %
Gamma_FCA=0.79;
V_FCA=6.9e-19;
Gamma_PhC=0.769;
tau_FC=150e-12;
tau_th=9.7e-9;
Gamma_m=2*pi*110e3; %1e6; % %5.51E06;

V_PhC=1e-18;
% kappa_fb=0; %1e4;
% kappa_inj=1e-13;

% 
omega0=2*pi*c/lambda0;
omegaL=2*pi*c/lambdaL;
del_omega=-abs(omega0-omegaL);

% Coupled differential equations

Udot=1i*(-g0*sqrt(2*m_eff*Omega_m/hbar)*X+omega0/nSi*(dnsi_dT*DT+dnsi_dN*N)+del_omega)*U-1/2*(gamma_i+gamma_e+Gamma_TPA*beta_si*c^2/(V_TPA*nSi^2)*abs(U)^2+sigma_si*c/nSi*N)*U+sqrt(gamma_e*(Pin+kappa_inj*E_inj+kappa_fb*abs(E_tau)^2));
Ndot =  -N/tau_FC + Gamma_FCA*beta_si*c^2/(2*hbar*omega0*V_FCA^2*nSi^2)*abs(U)^4;
DTdot =-DT/tau_th+Gamma_PhC/(rho_si*Cp*V_PhC)*(gamma_i+Gamma_TPA*beta_si*c^2/(V_TPA*nSi^2)*abs(U)^2+sigma_si*c/nSi*N)*abs(U)^2;
Xdot=Xprime;
Xprimedot=-Gamma_m*Xprime-Omega_m^2*X+g0/omega0*sqrt(2*Omega_m/(hbar*m_eff))*abs(U)^2;


dydt = [real(Udot); imag(Udot);Ndot; DTdot; Xdot; Xprimedot];
