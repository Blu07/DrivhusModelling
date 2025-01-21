




#%% Samle datapunkter for lys, temp-diff og temp endring inne.

# Modell for endring av inne-temp ut ifra lys inn og temp-diff (parametre for 3D-kurve-plan)
# Modell for lokale lysforhold (parametre for lys inn for hvert tidspunkt)


#%% Modell for lys-inn i lokale fohold
# Lag en liste med lister. Hver liste er én dag.
# Hver dag inneholder en liste med lysforoldene for hver spesifikke tid. f.eks. hver time.
# Lag en xListe med unix-tiden for antall sekunder siden 00:00:00
# Finn unix-tiden for 00:00:00 for DEN DAGEN
# Trekk denne verdien fra alle elementene i lista. 
# Interpoler lista med lys til å passe med xListen med unix-tider etter kl. 00:00:00.
# Legg den interpolerte lista til i lista for alle dager

# For å visualisere:
# Plot alle punktene.

# Lag en funksjon for en linje som passer gjennom lysforholdet for alle dagene på de spesifikke tidene.
# F.eks. en funksjon som passer for lysforholdet kl. 08:00.
# Siden solen går opp tidligere om sommeren, vil denne måten finne en fuksjon som passer for data som går gjennom alle tider i året.
# lagre parameterne for funksjonen i hvert klokkeslett.
# Gi dette til PROGRAM 2.




#%% MODELL for endring i temp inne ut ifra sol inn og temp diff:


# FAKTISK LYS INN
# Enkelt: datapunktet for det tidspunktet, evt. interpolasjon


# FAKTISK TEMP DIFF
# Enkelt: ta diff av temp ute og inne for det tidspunktet, evt diff mellom interpolasjon.  

# 



# FAKTISK ENDRING I TEMP INNE:
# Enkelt: Finn stigningstall for lineær regresjon gjennom en haug med punkter rundt tidspunktet.
# Funn ut hvor mange punkter gir best resultat.
# Husk:
    # 5 punkter per intervall.
    # Bruk en faktor på 5 
# Kanksje 3*5? Da blir punktene rundt det tidspunktet brukt, samt intervallet før og etter.
# Kanskje et par andre også, men det har sikkert ikke så stor effekt

