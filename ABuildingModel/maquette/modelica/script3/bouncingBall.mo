model boucingBall
    constant Real g=9.81 ;
    //c est le coefficient d'amortissement au rebond
    parameter Real c=0.9 ;
    
    //h est la hauteur de la balle
    parameter Real h0=1;
    Real h(start=h0);
    
    //v est la vitesse de la balle
    Real v(start=0);

equation
    der(h)=v;
    der(v)=-g;
    when h<=0 and v<0 then
        reinit(v,-c*pre(v));
        reinit(h,0);
    end when;
    /*
    pour que la balle ne tombe pas dans le vide ?
    */
    if h<0 then
        h=0;
    end if;
  
end boucingBall;