model Hysteresis2
    Real T(start=10);
    Real Consigne=20;
    Real derivate;
    Boolean heating;
    
initial equation
    heating = if T<Consigne then true else false;

equation
    der(T)=derivate;
    
algorithm
    when {T>=Consigne+2, T<=Consigne-1} then
        heating := T<=Consigne;
    end when;
    if heating then
        //chauffage en marche
        derivate:=5-T/5;
    else
        //arrêt du chauffage
        derivate:=-T/10;        
    end if;
    
end Hysteresis2;