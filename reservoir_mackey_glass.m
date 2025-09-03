%% Call reservoir
clear all;
c=299792458;
lambda0= 1572.8e-9  ; %optical cavity resonance
Pin=1e-3;
Id=sqrt(6.56e20);
kappaext=2*pi*5.00E09; %2*pi*2.2e9;
kappa_inj=4e-14;
kappa_fb=0;

dt=1e-9; 
dt1=1e-11;
num_nodes=50; 
modmax=0.5; 
DTcoeff=0.1; 
del0=4; % detuning in multiples of gamma_i 
gamma_i=19e9; 
omega0=2*pi*c/lambda0;
omegaL=omega0+del0*gamma_i;
lambdaL=2*pi*c/omegaL;

% prepare data 
init_length = 1;
train_length =  2000;
test_length = 1000;
max_pred_steps=5;

mg_t = readmatrix("mackey_glass_tau17.csv");

[y_res]=fn_reservoir_mackey_glass(mg_t, modmax,dt,dt1, y0, Pin,lambdaL, kappa_fb, kappa_inj, Id, num_nodes, max_pred_steps, init_length,train_length,test_length,DTcoeff);                
y_res_train=y_res(init_length*num_nodes+1:(init_length+train_length)*num_nodes,:);
y_res_test=y_res((init_length+train_length)*num_nodes+1:end,:);
states_train=reshape(y_res_train(:,1).^2+y_res_train(:,2).^2, num_nodes, train_length) ;
pred_steps_array=linspace(-max_pred_steps,max_pred_steps,2*max_pred_steps+1);
NRMSE_test=zeros(1,2*max_pred_steps+1);
for j=1:2*max_pred_steps+1
    pred_step=pred_steps_array(j);
    y_t = mg_t(end-max_pred_steps+pred_step-input_length+1:end-max_pred_steps+pred_step);
    y_train = y_t(init_length + 1:init_length + train_length);
    y_test = y_t(init_length + train_length + 1:init_length + train_length + test_length);
    w_train = states_train' \ y_train;
    states_test=reshape(y_res_test(:,1).^2+y_res_test(:,2).^2,  num_nodes, test_length) ;
    y_pred_test=w_train'*states_test;
    NRMSE_test(j)=sqrt(1/test_length*sum((y_pred_test' - y_test).^2/var(y_test)));
end
