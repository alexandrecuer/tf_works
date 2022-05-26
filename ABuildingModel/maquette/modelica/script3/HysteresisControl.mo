model HysteresisControl 
/*
The most basic control strategy
*/
    Real T(start=30);
    Boolean heat;
    Real Q;
    Real Consigne=20;
    
initial equation
    heat = if T<Consigne then true else false;

equation    
    Q = if heat then 25 else 0;
    /*
    2 est une sorte de coefficient de transfert de chaleur en W/K
    */
    der(T) = Q-2*(T-10);
    
algorithm
    /*
    on ne fait rien dans l'intervalle Consigne-1 à Consigne+1 :
       - si on est en train de chauffer, on continue
       - si on ne ne chauffe plus, on continue
    on ne met à jour le booléen heat que lorsqu'on est hors de cet intervalle
    */
    // option 1
    /*
    when T>Consigne+1 then
      heat := false;
    end when;
    when T<Consigne-1 then
      heat := true;
    end when;
    */
    // option 2
    when T>Consigne+1 or T<Consigne-1 then
      heat := T<=Consigne;
    end when;
    
end HysteresisControl;