

%Set initial values
y0=[0,0,0,0,0,0];
y_tau=0;
DTcoeff=[0.3]; %,0.2,0.13,0.1,3]; % 0-OMO, 0.2-FOMO/2, 0.13-FOMO/3, 0.1-FOMO/4 0.3 - chaos
dt=1e-9; % 0.5e-9, 1e-9, 2e-9, falta 2e-9, 200 nodes
num_nodes=50;

Pin=1e-3;
c=299792458;
lambda0= 1572.8e-9  ; %optical cavity resonance
y=y0;
Id=sqrt(6.56e20);

kappaext=2*pi*2.2e9;
delay=1;
dt1=1e-11;
kappa_inj=[4e-14]; %value range 1e-16 to 1e-13

Pin=1e-3;
kappa_fb=0; %0.8e9; %[0, 1e3, 5e3, 1e4 ];
del0=4; % detuning in multiples of gamma_i 
gamma_i=19e9; 
omega0=2*pi*c/lambda0;
omegaL=omega0+del0*gamma_i;
lambdaL=2*pi*c/omegaL;
modmax=0.5;


%Load data
load("spokendigits.mat");


% Prepare data 
init_length=1;
train_length = 399;
test_length = 100; 

n_freq_channels= 86;
n_classifiers=10;

input_length = init_length+train_length + test_length;
neworder=randperm(input_length);

%% Run the reservoir
options = odeset('RelTol',1e-3,'AbsTol',1e-6, 'Stats','off');


% Train and test data - parallel
for i=1:init_length
    spd_inputs{i}= spd.data(neworder(i)).inputs;
end
spd_t_init=cell2mat(spd_inputs(1:init_length));
spd_t_init=((spd_t_init-min(spd_t_init))./(max(spd_t_init)-min(spd_t_init))-0.5)*2;

mask= (rand(num_nodes, n_freq_channels) -0.5)*2;
driving_sig_masked_init=spd_t_init'*mask';
driving_sig_masked_resh_init=reshape(driving_sig_masked_init', [],1);
for i_sig=1:length(driving_sig_masked_resh_init)     
    E_inj=Id*driving_sig_masked_resh_init(i_sig);
    ode=@(t,y) fn_coupledsystem_diff_equations(t,y, Pin*modmax, lambdaL,E_inj,y_tau, kappa_fb,  kappa_inj*(1-modmax), DTcoeff);    
    [t,y] = ode15s(ode, [(i_sig)*dt:dt1:(i_sig+1)*dt], y, options);   
    y=y(end,:);
end
y0_init=y;
parfor i=init_length+1:init_length+train_length+test_length     
    spd_t= spd.data(neworder(i)).inputs;
    
    driving_sig_masked=spd_t'*mask';
    driving_sig_masked_resh=reshape(driving_sig_masked', [],1);

    y=y0_init;
    y_res=zeros(length(driving_sig_masked_resh),6);
    for i_sig=1:length(driving_sig_masked_resh)
        % i_sig 
        E_inj=Id*driving_sig_masked_resh(i_sig);
       
        ode=@(t,y) fn_coupledsystem_diff_equations(t,y, Pin*modmax, lambdaL,E_inj,y_tau, kappa_fb,  kappa_inj*(1-modmax), DTcoeff);
        
        [t,y] = ode15s(ode, [(i_sig)*dt:dt1:(i_sig+1)*dt], y, options);   
        y_res(i_sig,:)=y(end,:);
        y=y(end,:);
           
    end
    s = struct("y_res",y_res);
    save( sprintf("yres_%d_theta_%d_nodes_%d_DTCoeff_%d.mat",i, dt,num_nodes,DTcoeff),"-fromstruct",s);

end


%%
y_res_all=[];

for i=init_length+1:init_length+train_length+test_length     
    load( sprintf("yres_%d_theta_%d_nodes_%d_DTCoeff_%d.mat",i, dt,num_nodes,DTcoeff));
    y_res_all=[y_res_all;y_res];
end

L_sig=length(y_res_all);
transmission=y_res_all(:,1).^2+y_res_all(:,2).^2;
states=reshape(transmission, num_nodes,[])';
n_timesteps=zeros(input_length,1);
for i=1:input_length
    n_timesteps(i)= size(spd.data(neworder(i)).outputs,2);
    spd_outputs{i}= spd.data(neworder(i)).outputs;
end

st_train=1;
end_train=sum(n_timesteps(init_length+1:init_length+train_length));
st_test=end_train+1;
end_test=sum(n_timesteps(init_length+1:init_length+train_length+test_length));
states_train=states(st_train:end_train,:);
states_test=states(st_test:end_test,:);


output_train=cell2mat(spd_outputs(init_length+1:train_length+init_length));
output_test=cell2mat(spd_outputs(init_length+train_length+1:test_length+train_length+init_length));
output_train(output_train==-1)=0;
y_train=output_train;

w_out_train = states_train \ y_train';
y_pred_train= w_out_train'*states_train';


y_pred_test= w_out_train'*states_test';


n_timesteps_test=n_timesteps(init_length+train_length+1:train_length+init_length+test_length);
digit_pred_array_test=zeros(test_length,2);
for i=1:test_length
    if i==1
        stSig=1;
    else 
        stSig=sum(n_timesteps_test(1:i-1))+1;
    end
    endSig=sum(n_timesteps_test(1:i));
    digit_pred_array_test(i,1)=find(output_test(:, stSig:endSig)==1,1);
    y_pred_1dgt=y_pred_test(:,stSig:endSig);
    [maxval, digit]=max(sum(y_pred_1dgt,2));

    digit_pred_array_test(i,2)=digit;
end

WER=1-sum(digit_pred_array_test(:,1)==digit_pred_array_test(:,2))/test_length;
