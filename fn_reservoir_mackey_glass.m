function [y_res]=fn_reservoir_mackey_glass(mg_t, modmax,dt,dt1, y0, Pin,lambdaL, kappa_fb, kappa_inj, Id, num_nodes, max_pred_steps, init_length,train_length,test_length,DTcoeff)
    y=y0;
    
    mask = (rand(num_nodes, 1)'-0.5)*2 ;
   
    mg_t=((mg_t-min(mg_t))/(max(mg_t)-min(mg_t))-0.5)*2;

    y_tau=y(1)+1j*y(2);
        
   
    options = odeset('RelTol',1e-3,'AbsTol',1e-6, 'Stats','off');
   
    input_length = init_length + train_length + test_length;
    
    u_t = mg_t(end-max_pred_steps-input_length+1:end-max_pred_steps);
  
    driving_sig=reshape((u_t*mask)',[],1);
    y_res=zeros(length(driving_sig),6);

    for i=1:length(driving_sig)
        E_inj=Id*driving_sig(i);
        ode=@(t,y) fn_coupledsystem_diff_equations(t,y, Pin*modmax, lambdaL,E_inj,y_tau, kappa_fb, kappa_inj*(1-modmax),DTcoeff);
        [t,y] = ode15s(ode, [i*dt:dt1:(i+1)*dt], y, options); 
        y_res(i,:)=y(end,:);
        y=y(end,:);
    end
end

