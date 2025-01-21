


# Simulere hver dag

# bruke modell for forhold mellom temp-diff og lys-inn for å finne temperaturendring




# Prognose for lys-inn lages ved å sammenlikne:
# - beregnet solinnstråling
# - forventet skydekke (gjennomsnittlig for sesong eller meldt vær fra MET)
# - modell for lys i lokale forhold (noen tidspunkt hvor lys inn endres drastsik på grunn av skygger?)
    # Fra PROGRAM 1


# Prognose for temp-ute kommer fra
# - modell for gjennomsnittlig temperatur for måneden målt de siste årene.
# - Trekker også inn målte temperaturer for å justere modellen til hensyn for lokale, målte forhold.
# - F.eks sier modellen at temperaturen har gjennomsnittlig vært 15.7C,
#   men den har vært 13.9C siste uka. Da kan modellen justeres ned.







# resulterer i flere datapunkter for hver dag:
# (en liste av dager med:)
# - høyeste temperatur på dagen
# - laveste temperatur på natten (trekkes fra prognose av utetemperatur, fordi drivhuset simuleres ikke over natta. Da er det ikke noe sollys, og temperaturen synker sakte til samme temperatur som ute.)
# - mediantemperatur på dagen
# - lengde med mediantemperatur +- 0.5 gjennom hele dagen (finn ut om 0.5 eller annen temperatur er best å måle) (kan avbrytes og gjenopptas, feks at temperaturen stiger litt over midt på dagen, og så kommer ned til median igjen.)

# - høyeste lysinnstråling på dagen
# - medianlysinnstråling på dagen
# - 

# - nattelengde (trekkes fra prognose av lys)
# - daglengde (trekkes fra prognose for lys)
# - 
