# les radiateurs

https://apalis.fr/herve.silve/radiateur.htm

## Transmission par rayonnement

La transmission thermique par rayonnement s'effectue par rayonnement électromagnétique et n'a pas besoin de matière pour se propager (le rayonnement solaire). La quantité de chaleur transmise par rayonnement est calculée avec la formule suivante :
```
Qr = S x hr x (Tm - Ti)
```
S est la surface d'échange thermique de l'émetteur, en m²

Tm est la température moyenne de l'émetteur, en °C

Ti est la température ambiante de la pièce, en °C

hr est le coefficient de transmission thermique par rayonnement, en W/(m².K) et s'obtient de la façon suivante :
```
hr = Ec x (T1 + T2) x (T1**2 + T2**2) x Co
```
Ec est l'émissivité de la surface qui vaut 1 pour un corps noir et qui est comprise entre 0 et 1 selon l'état de surface du matériau. Dans les locaux d'habitation les surfaces peuvent être considérées comme des corps noirs et la valeur de Ec est peu différente de l'unité mais la valeur moyenne pour Ec peut être prise égale à 0,9.

T1 et T2 sont les températures absolues, en K, des corps en présence, T1 pour la température moyenne de l'émetteur et T2 pour la température du local car on part de l'hypothèse que toutes les parois et objets sont à la même température que la température ambiante du local.

Co est la constante de Stefan-Boltzmann, et vaut 5,67051 x 10-8 `W/(m**2 K**4)`

## Transmission par convection

La transmission thermique par convection s'effectue par un mouvement des molécules d'air. Pour le cas des radiateurs, la convection est naturelle (libre) et l'échange de chaleur est responsable de ce mouvement. C'est le transfert de chaleur qui provoque le mouvement de ces molécules par la différence de densité qui est fonction de la température.

La quantité de chaleur transmise par convection peu être calculée avec la formule suivante :
```
Qc = S x hc x (Tm - Ti)
```

hc est le coefficient de transmission thermique par convection, en W/(m².K) et peut être obtenu, pour un radiateur classique, avec la formule empirique suivante :

```
hc = 5,6 x ((T1 - T2) / (T2 x h))**0,25
```

h est la hauteur de la surface de chauffe, en m

# calculer les émissions thermiques totales d'un radiateur (Qt).

La surface d'échange thermique de ce radiateur est de 2,5 m², sa hauteur est de 0,85 m, la température moyenne du fluide caloporteur est de 70 °C et la température du local est de 20 °C :

```
hr = 0,9 x (343,15 + 293,15) x (343,15**2 + 293,15**2) x (5,67051 x 10-8) = 6,614 W/(m².K)
Qr = 2,5 x 6,614 x (70 - 20) = 826,75 W
hc = 5,6 x ((343,15 - 293,15) / (293,15 x 0,85))**0,25 = 3,748 W/(m².K)
Qc =  2,5 x 3,748 x (70 - 20) = 468,50 W
Qt = 826,75 + 468,50 = 1295,25 W
```
