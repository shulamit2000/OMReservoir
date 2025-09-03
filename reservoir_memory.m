%% Call reservoir
clear all;
c=299792458;
lambda0= 1572.8e-9  ; %optical cavity resonance
y0=[0,0,0,0,0,0]; 
Pin_array=1e-3;
y=y0;
Id=sqrt(6.56e20);

kappaext=2*pi*5.00E09; %2*pi*2.2e9;

dt=1e-9;
dt1=[1e-11];

num_nodes=50;
modmax=0.5;
DTcoeff=0; %[0,0.1,0.2,0.13,3]; 
kappa_inj=4e-14;
kappa_fb=0; 
del0=4; % detuning in multiples of gamma_i 
gamma_i=19e9; 
omega0=2*pi*c/lambda0;
omegaL=omega0+del0*gamma_i;
lambdaL=2*pi*c/omegaL;

init_length = 1;
train_length =  2000;
test_length = 1000;
input_length=init_length+train_length+test_length;

max_pred_steps=5;
mg_t_mem=rand(500,1);

[y_res_mem]=fn_reservoir_memory_V2(mg_t_mem, modmax,dt,dt1, y0, Pin,lambdaL, kappa_fb, kappa_inj, Id, num_nodes, max_pred_steps, init_length,train_length,test_length,DTcoeff);                % parsave("results.mat","a","-append");

u_t = mg_t_mem_norm(end-max_pred_steps-input_length+1:end-max_pred_steps);


pred_steps_array=linspace(-max_pred_steps,-1,max_pred_steps);
MC=zeros(1,max_pred_steps);
for j=1:max_pred_steps
   
    pred_step=pred_steps_array(j);
    train_length_mem=train_length+pred_step;
    y_res_test=y_res(end-(test_length)*num_nodes+1:end,:);
    y_res_train=y_res(end-(test_length+train_length_mem)*num_nodes+1:end-(test_length)*num_nodes,:);

    states_train=reshape(y_res_train(:,1).^2+y_res_train(:,2).^2, num_nodes, train_length_mem) ;
    y_t = u_t(1:end+pred_step);
    y_test = y_t(end-(test_length)+1:end);
    y_train = y_t(end-(test_length+train_length_mem)+1:end-(test_length));

    w_train = states_train' \ y_train;
    states_test=reshape(y_res_test(:,1).^2+y_res_test(:,2).^2,  num_nodes, test_length) ;
    y_pred_test=w_train'*states_test;
    MC(j)=(corr(y_test,y_pred_test'));
end

