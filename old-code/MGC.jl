module MGCLaguerre


# Un objet representant une distirbution GGC :
# avec les méthodes associées correspondantes, qui cole avec le packet Distributions.
# Typiquement, cet object doit hériter des distributions et implementer les méthodes correspondantes.
# Il doit donc avoir des alpha, des scales, et un niveau de précision m par dimension. On fait les précalcul correspondants a ce niveau de précision à l'initialisation de l'objet.
# Cet objet peut avoir une méthode pour sortir la densitée de moshopoulos si on est en dimension 1
# et une méthode pour sortir la densitée de laguerre en dimension suppérieure, du coup en précisant le $m$.

# On peut également disigner la classe de distribution pour qu'elle se convolue facilement.
# c'est a dire définir la méthode + sur cette classe, et même overrider la méthode + sur les loies gammas du package Distibutions.

# Il faut aussi pouvoir utiliser mon algorithme, c'ets a dire fournir une fonction qui estime une GGC sur des coefficients de laguerre pré-fournis
# Elle peut directement contenir les précalculs. Mais je ne suis pas sur que rentrer dans une fonction le code d'estimation par particleSwarm st très malin: on pourrais l'estimer par autre chose.
# Il faut aussi une fonction qui pré-calcule ces coefficients, soit depuis une distributions du module Distribution, soit depuis un échantillon.






# End of the module :
end
