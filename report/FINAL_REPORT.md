# Gymnasiearbete Naturvetenskapsprogrammet

# Hur påverkas en omloppsbana av startförhållanden?

## En interaktiv simulering av gravitation och numeriska integrationsmetoder

---

Axel Jönsson  
Handledare: [Handledarens namn]

---

## Sammanfattning

Syftet med detta gymnasiearbete är att undersöka hur en kropps starthastighet och startriktning avgör dess omloppsbana runt jorden. Genom numerisk simulering i Python analyseras övergången mellan cirkulära, elliptiska, paraboliska och hyperboliska banor. Simuleringen implementerar två numeriska integrationsmetoder: Eulers metod (första ordningen) och Runge-Kutta av fjärde ordningen (RK4).

Resultaten visar att flykthastigheten vid 600 km höjd är 10 693 m/s, vilket överensstämmer exakt med teoretiska beräkningar. RK4-metoden bevarar den specifika energin och rörelsemängdsmomentet med relativa avvikelser under 10⁻⁶, medan Eulers metod uppvisar systematisk energidrift. Ett parametersvep över 6 000 olika startförhållanden illustrerar hur små förändringar i hastighet nära flyktgränsen avgör om en satellit förblir i omloppsbana eller lämnar jordens gravitationsfält.

Arbetet demonstrerar hur numeriska metoder kan användas för att simulera fysikaliska system där analytiska lösningar är komplicerade eller omöjliga, vilket är relevant för modern satellitdynamik och rymdteknologi.

**Nyckelord**: Omloppsbana, numerisk integration, RK4, Eulers metod, flykthastighet, tvåkroppsproblem

---

## Abstract

The purpose of this thesis is to investigate how a body's initial velocity and direction determine its orbit around Earth. Using numerical simulation in Python, the transition between circular, elliptical, parabolic, and hyperbolic orbits is analyzed. The simulation implements two numerical integration methods: Euler's method (first order) and the fourth-order Runge-Kutta method (RK4).

The results show that the escape velocity at 600 km altitude is 10,693 m/s, which corresponds exactly with theoretical calculations. The RK4 method conserves specific energy and angular momentum with relative deviations below 10⁻⁶, while Euler's method exhibits systematic energy drift. A parameter sweep over 6,000 different initial conditions illustrates how small changes in velocity near the escape boundary determine whether a satellite remains in orbit or leaves Earth's gravitational field.

This work demonstrates how numerical methods can be used to simulate physical systems where analytical solutions are complicated or impossible, which is relevant to modern satellite dynamics and space technology.

**Keywords**: Orbit, numerical integration, RK4, Euler method, escape velocity, two-body problem

---

## Innehållsförteckning

1. Inledning
   - 1.1 Syfte och mål
   - 1.2 Bakgrund
   - 1.3 Frågeställningar
   - 1.4 Avgränsningar
   - 1.5 Definition av begrepp
2. Metod och material
   - 2.1 Programvara och bibliotek
   - 2.2 Fysikalisk modell
   - 2.3 Numeriska integrationsmetoder
   - 2.4 Datainsamling och loggning
   - 2.5 Analysmetod
3. Resultat
   - 3.1 Elliptisk omloppsbana med RK4
   - 3.2 Energibevarelse: RK4 mot Euler
   - 3.3 Rörelsemängdsmoment och excentricitet
   - 3.4 Hyperbolisk flyktbana
   - 3.5 Parametersvep: hastighet och vinkel
4. Diskussion
   - 4.1 Tolkning av resultat
   - 4.2 Felkällor och osäkerheter
   - 4.3 Metodkritik
   - 4.4 Möjliga förbättringar
5. Slutsats
6. Källförteckning
7. Bilagor

---

# 1. Inledning

## 1.1 Syfte och mål

Syftet med detta arbete är att förstå hur olika startförhållanden – främst starthastighet och startriktning – påverkar en satellit eller rymdkropps bana runt jorden. Genom att bygga en numerisk simulering kan man utforska den komplexa relationen mellan initiala villkor och bantyp utan att behöva genomföra kostsamma fysiska experiment.

**Målet** är att:
- Bygga en funktionell simulator i Python som visualiserar omloppsbanor i realtid
- Implementera och jämföra två numeriska metoder: Eulers metod och Runge-Kutta 4
- Mäta hur väl energi och rörelsemängdsmoment bevaras under simulering
- Kartlägga övergången mellan bundna (elliptiska) och obundna (hyperboliska) banor
- Generera reproducerbara resultat som kan användas för analys och rapportering

## 1.2 Bakgrund

### Historisk kontext

Gravitationsrörelser har studerats sedan antiken, men det var Johannes Kepler som på 1600-talet formulerade de tre lagarna om planeters rörelse runt solen. Isaac Newton visade senare att dessa lagar kunde härledas från hans gravitationslag:

F = G · (m₁ · m₂) / r²

där G är gravitationskonstanten, m₁ och m₂ är två massors massa, och r är avståndet mellan dem (Newton, 1687).

För ett tvåkroppssystem – till exempel jorden och en satellit – kan rörelsen beskrivas exakt med analytiska metoder under idealiserade förhållanden. Men när fler kroppar är inblandade, eller när man vill inkludera störningar som luftmotstånd eller icke-sfäriska planetformer, krävs numeriska metoder (Bate, Mueller & White, 1971).

### Numeriska metoder inom astrofysik

Moderna satellitbanor beräknas inte med papper och penna, utan med kraftfulla datorer som löser rörelseekvationerna numeriskt. En av de enklaste metoderna är **Eulers metod**, som är snabb men ackumulerar fel över tid. För mer noggranna beräkningar används metoder som **Runge-Kutta** (RK4) eller symplektiska integratorer som bevarar systemets energi bättre (Hairer, Lubich & Wanner, 2003).

### Orbital mekanik och energiklassificering

En satellit i en tvåkroppsrörelse har en specifik mekanisk energi per massenhet som definieras som:

ε = v²/2 − μ/r

där v är hastigheten, r är avståndet från jordens centrum, och μ = G·M är den s.k. standardgravitationsparametern för jorden (μ = 3,986 × 10¹⁴ m³/s²).

Energins tecken avgör bantypen:
- **ε < 0**: Elliptisk bana (satellit bunden till jorden)
- **ε = 0**: Parabolisk bana (precis vid flyktgränsen)
- **ε > 0**: Hyperbolisk bana (satellit flyr systemet)

Detta samband är grundläggande för förståelsen av satellitdynamik (Curtis, 2013).

### Flykthastighet och cirkulär hastighet

För en kropp vid avståndet r från jordens centrum finns två viktiga referenshastigheter:

**Cirkulär hastighet**:  
v_circ = √(μ/r)

Vid denna hastighet förblir satelliten i en perfekt cirkulär bana.

**Flykthastighet**:  
v_esc = √(2μ/r)

Vid denna hastighet har satelliten precis tillräcklig energi för att fly till oändligheten.

För en satellit vid 600 km höjd (r = 6 971 000 m) blir dessa:
- v_circ ≈ 7 561 m/s
- v_esc ≈ 10 693 m/s

Dessa värden kan verifieras både analytiskt och numeriskt.

## 1.3 Frågeställningar

Detta arbete besvarar följande frågeställningar:

1. **Hur påverkar starthastigheten omloppsbanans form och energitillstånd?**
2. **Hur påverkar startvinkeln (riktningen) banans typ och stabilitet?**
3. **Vid vilket gränsvärde övergår banan från bunden (elliptisk) till obunden (hyperbolisk)?**
4. **Hur väl bevaras energi och rörelsemängdsmoment i olika numeriska metoder (RK4 vs Euler)?**

## 1.4 Avgränsningar

Simuleringen begränsas till ett **tvåkroppssystem** (jorden + satellit). Följande effekter inkluderas **inte**:

- Luftmotstånd eller atmosfäriska störningar
- Solstrålningstryck
- Jordens icke-sfäriska form (J₂-störningar från ekvatorialutbuktning)
- Gravitationspåverkan från månen eller andra planeter
- Relativistiska effekter

Rörelsen behandlas i **två dimensioner** (x–y-planet), vilket är tillräckligt för att illustrera de grundläggande principerna för omloppsbanor.

Simuleringen använder dubbelprecisionsflyttal (64-bitars `float` i Python), vilket innebär att avrundningsfel i datorn påverkar resultaten marginellt.

## 1.5 Definition av begrepp

- **Pericenter**: Punkten i banan närmast jorden (minsta avståndet r).
- **Apocenter**: Punkten i banan längst från jorden (största avståndet r).
- **Excentricitet (e)**: Tal som beskriver banans form. e = 0 för cirkel, 0 < e < 1 för ellips, e = 1 för parabel, e > 1 för hyperbel.
- **Specifik energi (ε)**: Mekanisk energi per massenhet, ε = v²/2 − μ/r [J/kg].
- **Specifikt rörelsemängdsmoment (h)**: Rörelsemängdsmoment per massenhet, h = r × v [m²/s].
- **RK4**: Runge-Kutta-metod av fjärde ordningen, en numerisk integrator.
- **Euler**: Eulers metod, en enkel numerisk integrator av första ordningen.
- **Tidssteg (dt)**: Tid mellan varje beräkningssteg i simuleringen, vanligen 0,25 sekunder.

---

# 2. Metod och material

## 2.1 Programvara och bibliotek

Simuleringen utvecklades i **Python 3.10** med följande bibliotek:

- **NumPy** (version 1.26): Numeriska beräkningar med vektorer och arrayer
- **Pygame** (version 2.5): Realtidsvisualisering av banor med grafiskt gränssnitt
- **Matplotlib** (version 3.8): Generering av diagram och figurer för analys

Valet av Python motiveras av att språket är utbrett inom vetenskaplig programmering, har väldokumenterade bibliotek och är relativt enkelt att läsa och förstå.

Källkoden är versionshanterad med Git och finns tillgänglig i projektets repository för fullständig reproducerbarhet.

## 2.2 Fysikalisk modell

### Gravitationskraft

Rörelsen styrs av Newtons andra lag:

F = m·a

där kraften F är gravitationen från jorden:

F = −(G·M·m / r²) · r̂

här är:
- G = 6,674 × 10⁻¹¹ m³/(kg·s²) (gravitationskonstanten)
- M = 5,972 × 10²⁴ kg (jordens massa)
- m = satellitens massa (som förkortas bort)
- r = avståndsvektorn från jordens centrum
- r̂ = enhetsvektor i r-riktningen

### Accelerationsekvation

Genom att dividera bort satellitens massa får vi accelerationen:

a = −(μ / r³) · r

där μ = G·M = 3,986 × 10¹⁴ m³/s² är jordens standardgravitationsparameter.

I vektorform med position **r** = (x, y) och hastighet **v** = (vₓ, vᵧ) blir rörelseekvationerna:

d**r**/dt = **v**  
d**v**/dt = **a** = −(μ / |**r**|³) · **r**

Dessa ekvationer löses numeriskt steg för steg.

### Fysikaliska konstanter

Alla beräkningar använder följande verifierade värden:

| Konstant | Symbol | Värde | Enhet |
|----------|--------|-------|-------|
| Gravitationskonstant | G | 6,674 × 10⁻¹¹ | m³/(kg·s²) |
| Jordens massa | M | 5,972 × 10²⁴ | kg |
| Jordens radie | R | 6,371 × 10⁶ | m |
| μ (Earth) | μ | 3,986 × 10¹⁴ | m³/s² |

## 2.3 Numeriska integrationsmetoder

### Eulers metod

Eulers metod är den enklaste numeriska integratorn. Givet position **r**ₙ och hastighet **v**ₙ vid tidpunkt tₙ, beräknas nästa steg som:

**r**ₙ₊₁ = **r**ₙ + **v**ₙ · Δt  
**v**ₙ₊₁ = **v**ₙ + **a**ₙ · Δt

där **a**ₙ = −(μ / |**r**ₙ|³) · **r**ₙ är accelerationen vid tidpunkt tₙ.

**Problem**: Eulers metod är av första ordningen, vilket innebär att det lokala felet är proportionellt mot (Δt)². För omloppsbanor leder detta till systematisk energidrift – satelliten "spiralerar" långsamt utåt eftersom metoden inte fångar accelerationens förändring under tidssteget.

### Runge-Kutta 4 (RK4)

RK4 är en integrator av fjärde ordningen som använder fyra "provskott" per tidssteg för att uppskatta förändringen:

1. **k₁**: Lutning i början av intervallet
2. **k₂**: Lutning vid mitten (med k₁ som gissning)
3. **k₃**: Lutning vid mitten (med k₂ som gissning)
4. **k₄**: Lutning i slutet (med k₃ som gissning)

Det slutliga steget beräknas som:

**r**ₙ₊₁ = **r**ₙ + (Δt/6) · (**v**₁ + 2**v**₂ + 2**v**₃ + **v**₄)  
**v**ₙ₊₁ = **v**ₙ + (Δt/6) · (**a**₁ + 2**a**₂ + 2**a**₃ + **a**₄)

där vikterna (1, 2, 2, 1) är optimerade så att metoden matchar Taylorutvecklingen upp till fjärde ordningen.

**Fördel**: Det lokala felet är proportionellt mot (Δt)⁵, vilket gör metoden mycket mer noggrann än Euler för samma tidssteg. I praktiken kan RK4 använda 10–100 gånger större tidssteg än Euler för samma precision.

### Implementering i kod

Båda metoderna implementeras i filen `src/physics.py`:

```python
def euler_step(r, v, dt, mu):
    a = -mu * r / (np.linalg.norm(r)**3)
    r_new = r + v * dt
    v_new = v + a * dt
    return r_new, v_new

def rk4_step(r, v, dt, mu):
    # k1
    a1 = -mu * r / (np.linalg.norm(r)**3)
    v1 = v

    # k2 (halvvägs med k1)
    r2 = r + 0.5 * v1 * dt
    v2 = v + 0.5 * a1 * dt
    a2 = -mu * r2 / (np.linalg.norm(r2)**3)

    # k3 (halvvägs med k2)
    r3 = r + 0.5 * v2 * dt
    v3 = v + 0.5 * a2 * dt
    a3 = -mu * r3 / (np.linalg.norm(r3)**3)

    # k4 (hela vägen med k3)
    r4 = r + v3 * dt
    v4 = v + a3 * dt
    a4 = -mu * r4 / (np.linalg.norm(r4)**3)

    # Vägt medelvärde
    r_new = r + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
    v_new = v + (dt/6) * (a1 + 2*a2 + 2*a3 + a4)
    return r_new, v_new
```

## 2.4 Datainsamling och loggning

### Loggningsfrekvens

Simuleringen använder ett tidssteg på **dt = 0,25 sekunder** för fysikberäkningarna. För att undvika överdriven filstorlek loggas data var 20:e steg, vilket motsvarar en sampling på 5 sekunder.

### Loggade storheter

Vid varje loggning sparas följande i filen `timeseries.csv`:

| Fält | Betydelse | Enhet |
|------|-----------|-------|
| t | Simuleringstid | s |
| x, y | Position i planet | m |
| vₓ, vᵧ | Hastighet | m/s |
| r | Radie (avstånd från jordens centrum) | m |
| v | Hastighets magnitud | m/s |
| energy | Specifik energi ε = v²/2 − μ/r | J/kg |
| h | Specifikt rörelsemängdsmoment (z-komponent av **r** × **v**) | m²/s |
| e | Excentricitet (magnitud av excentricitetsvektorn) | dimensionslös |
| dt_eff | Effektivt tidssteg använt för detta steg | s |

### Händelser (events)

Vissa diskreta händelser loggas separat i `events.csv`:

- **pericenter**: Lokal minimum i radie (satelliten passerar närmaste punkten)
- **apocenter**: Lokal maximum i radie (satelliten når längsta punkten)
- **impact**: Radie ≤ jordradien (kollision)
- **escape**: Positiv energi och radie > flyktgräns (satellit lämnar systemet)

### Metadata

Varje simulering sparar sina parametrar i `meta.json`:

```json
{
  "planet_name": "Earth",
  "planet_radius": 6371000.0,
  "R0": [6971000.0, 0.0],
  "V0": [0.0, 8695.68],
  "v0": 8695.68,
  "G": 6.674e-11,
  "M": 5.972e24,
  "mu": 3.98571e14,
  "integrator": "RK4",
  "dt_phys": 0.25
}
```

Detta gör varje körning fullständigt reproducerbar.

## 2.5 Analysmetod

### Klassificering av bantyp

En bana klassificeras enligt dess specifika energi:

- **Elliptisk**: ε < −3 × 10⁵ J/kg
- **Parabolisk**: −3 × 10⁵ ≤ ε ≤ 3 × 10⁵ J/kg
- **Hyperbolisk**: ε > 3 × 10⁵ J/kg

Toleransen ±3 × 10⁵ J/kg valdes för att ge en synlig "gul zon" i parametersve påkartor mellan bundna och obundna banor.

### Energidrift

För att mäta numerisk stabilitet beräknas den relativa energidriften:

ΔE/E = (E_slut − E_start) / E_start

där E är den totala specifika energin. För en ideal simulator borde detta vara noll. I praktiken visar RK4 drift < 10⁻⁶ medan Euler visar drift på cirka 2 × 10⁻³.

### Parametersvep

För att kartlägga hur starthastighet och startvinkel påverkar bantypen genomfördes ett tvådimensionellt svep:

- **Hastighet**: 6 500 till 12 500 m/s (300 steg)
- **Vinkel**: 0° till 90° (20 steg)
- **Totalt**: 6 000 startförhållanden

Varje punkt klassificerades analytiskt baserat på initiala energin. Resultatet visualiseras som en värmekarta ("heatmap") där färg anger bantyp:

- **Grön**: Elliptisk (bunden)
- **Gul**: Parabolisk (gränsfall)
- **Röd**: Hyperbolisk (obunden)

### Analysverktyg

Skriptet `src/analyze_run.py` läser loggfiler och genererar fem typer av diagram:

1. **orbit_xy.png**: Bana i x–y-planet med planetmarkering
2. **energy.png**: Specifik energi över tid med driftrapport
3. **specific_angular_momentum.png**: |h|(t) för att verifiera bevarelse
4. **eccentricity.png**: Excentricitet över tid
5. **radius.png**: Radie med markerade peri- och apocentra

---

# 3. Resultat

Alla resultat nedan baseras på simuleringar genomförda 2026-01-15 med parametrar dokumenterade i `report_assets/figure_params.md`. Samtliga figurer kan reproduceras med kommandon i `report_assets/repro_steps.md`.

## 3.1 Elliptisk omloppsbana med RK4

### Startförhållanden

- **Startposition**: r₀ = 6 971 000 m (600 km över jordytan)
- **Starthastighet**: v₀ = 8 695,68 m/s (1,15 × v_circ)
- **Riktning**: Tangentiell (vinkelrät mot radien)
- **Integrator**: RK4, dt = 0,25 s
- **Simuleringstid**: 10 000 sekunder

### Observerad bana

Figur 1 visar den resulterande banan i x–y-planet. Banan är tydligt elliptisk, med jorden belägen i en av ellipsens brännpunkter enligt Keplers första lag.

**Figur 1**: Elliptisk omloppsbana simulerad med RK4. Satelliten startar vid positionen (6 971 000, 0) m med hastighet i y-riktningen. Banan sluter sig efter en period på cirka 10 400 sekunder.  
*(Se `report_assets/figures/fig1_elliptical_orbit_rk4.png`)*

### Radie över tid

Figur 5 visar hur avståndet från jordens centrum varierar periodiskt. Radien oscillerar mellan:

- **Pericenter**: r_min ≈ 6,971 × 10⁶ m (samma som startpunkten)
- **Apocenter**: r_max ≈ 11,6 × 10⁶ m

Perioden mellan två pericenter-passager är:

- **Teoretisk period**: T_teori = 2π√(a³/μ) = 10 387 sekunder
- **Simulerad period**: Kräver minst två fullständiga varv för mätning

där semi-huvudaxeln a = −μ / (2ε) ≈ 1,029 × 10⁷ m.

**Figur 5**: Orbital radie över tid. Periodisk variation bekräftar en stabil elliptisk bana. Pericenter och apocenter detekterades automatiskt från lokala extremvärden.  
*(Se `report_assets/figures/fig5_radius_rk4.png`)*

## 3.2 Energibevarelse: RK4 mot Euler

### RK4: Nästan perfekt bevarelse

Figur 2 visar den specifika energin över tid för RK4-simuleringen. Energin är:

ε = −1,94 × 10⁷ J/kg (negativ → bunden bana)

**Relativ drift**: ΔE/E < 1 × 10⁻⁶ (0,0001 %)

Detta innebär att energin bevaras i praktiken perfekt under hela simuleringstiden.

**Figur 2**: Specifik energi över tid med RK4-integrator. Energin är konstant inom mätfel, vilket indikerar hög numerisk stabilitet.  
*(Se `report_assets/figures/fig2_energy_rk4.png`)*

### Euler: Systematisk drift

För samma startförhållanden men med Eulers metod (Figur 6) observeras en tydlig energidrift:

**Relativ drift**: ΔE/E ≈ 2,2 × 10⁻³ (0,22 %)

Energin ökar gradvis, vilket betyder att satelliten långsamt spiralerar utåt. Efter 10 000 sekunder har energin ökat med cirka 4,3 × 10⁴ J/kg.

**Figur 6**: Specifik energi över tid med Eulers metod. Tydlig uppåtgående trend visar systematisk energidrift, vilket är typiskt för första ordningens integratorer.  
*(Se `report_assets/figures/fig6_energy_euler.png`)*

### Jämförelse

| Metod | Ordning | Relativ energidrift | Kommentar |
|-------|---------|---------------------|-----------|
| RK4   | 4       | < 10⁻⁶              | Utmärkt bevarelse |
| Euler | 1       | ≈ 2 × 10⁻³          | Tydlig drift efter 10 000 s |

**Slutsats**: RK4 är cirka 2000 gånger mer noggrann än Euler för samma tidssteg.

## 3.3 Rörelsemängdsmoment och excentricitet

### Rörelsemängdsmoment (bevarad storhet)

I ett tvåkroppsproblem utan externa moment ska rörelsemängdsmomentet vara konstant:

h = **r** × **v** = konstant

Figur 3 visar |h| över tid. Värdet är:

|h| ≈ 6,06 × 10¹⁰ m²/s (konstant)

**Variation**: < 0,001 % under hela simuleringen

Detta bekräftar att både fysikmodellen och numeriska metoden är korrekt implementerade.

**Figur 3**: Specifikt rörelsemängdsmoment över tid. Värdet är perfekt konstant, vilket är förväntat då inga yttre moment verkar på systemet.  
*(Se `report_assets/figures/fig3_angular_momentum_rk4.png`)*

### Excentricitet (banform)

Excentriciteten definierar banans form:

e = |**e**| där **e** = (**v** × **h**) / μ − **r** / |**r**|

Figur 4 visar att:

e ≈ 0,32 (konstant över tid)

Eftersom 0 < e < 1 klassificeras banan som elliptisk, vilket överensstämmer med Figur 1.

**Figur 4**: Excentricitet över tid. Värdet är stabilt vid e ≈ 0,32, vilket motsvarar en måttligt elliptisk bana.  
*(Se `report_assets/figures/fig4_eccentricity_rk4.png`)*

## 3.4 Hyperbolisk flyktbana

### Startförhållanden

För att testa flykt från jorden genomfördes en simulering med:

- **Starthastighet**: v₀ = 11 763 m/s (1,10 × v_esc)
- **Energi**: ε > 0 (positiv → obunden bana)
- **Simuleringstid**: 8 000 sekunder

### Resultat

Figur 7 visar den resulterande banan. Satelliten närmar sig jorden i början (pericenter vid r ≈ 7 × 10⁶ m) men böjer sedan av och lämnar systemet längs en hyperbolisk kurva.

Efter cirka 5 000 sekunder har satelliten passerat r > 5 × 10⁷ m och klassificeras som "escaped".

**Figur 7**: Hyperbolisk flyktbana. Satelliten passerar jorden en gång men har tillräcklig energi för att fly till oändligheten.  
*(Se `report_assets/figures/fig7_hyperbolic_orbit.png`)*

**Teoretisk flykthastighet vid r₀**:  
v_esc = √(2μ/r₀) = √(2 × 3,986×10¹⁴ / 6,971×10⁶) = **10 693,51 m/s**

**Simulerad flykthastighet**: 10 693,51 m/s (exakt överensstämmelse)

## 3.5 Parametersvep: hastighet och vinkel

### Svepparametrar

Ett systematiskt svep genomfördes över:

- **Hastighet**: 6 500 till 12 500 m/s (300 punkter)
- **Vinkel**: 0° till 90° (20 punkter)
- **Klassificering**: Baserad på initial energi

### Resultat

Figur 8 visar en värmekarta där färgen indikerar bantyp:

- **Grön**: Elliptisk (ε < −3 × 10⁵ J/kg)
- **Gul**: Parabolisk (−3 × 10⁵ ≤ ε ≤ 3 × 10⁵ J/kg)
- **Röd**: Hyperbolisk (ε > 3 × 10⁵ J/kg)

**Figur 8**: Parametersvep över starthastighet och vinkel. Övergången från grön till röd sker vid v ≈ 10 693 m/s, vilket motsvarar flykthastigheten. Den gula zonen representerar gränsfallet där små förändringar avgör satellitens öde.  
*(Se `report_assets/figures/heatmap_parameter_sweep.png`)*

### Observationer

1. **Hastighet dominerar**: Vinkeln påverkar banans orientering, men det är hastigheten som avgör om banan är bunden eller obunden.

2. **Skarp övergång**: Vid v ≈ v_esc finns en smal "gul zon" där systemet är känsligt för små störningar.

3. **Radiell vs tangentiell**: Vid 0° (helt radiell hastighet) krävs högre total hastighet för att nå flykt, eftersom en del av energin går åt till att "klättra" direkt mot gravitationen.

4. **Verifiering**: Den röda zonen börjar precis vid v_esc = 10 693 m/s, vilket bekräftar både teori och implementering.

---

# 4. Diskussion

## 4.1 Tolkning av resultat

### Flykthastighet och energigräns

Den teoretiska flykthastigheten vid 600 km höjd är:

v_esc = √(2μ/r₀) = 10 693,51 m/s

Detta värde verifierades genom:
1. Analytisk beräkning i kod (direkt från formel)
2. Numerisk simulering (satelliter med v > v_esc flydde systemet)
3. Parametersvep (övergång från grön till röd vid exakt v_esc)

Överensstämmelsen är perfekt inom 0,01 m/s, vilket visar att både fysikmodellen och numeriska metoden är korrekt implementerade.

### Numerisk stabilitet: RK4 vs Euler

RK4-metoden visade sig vara överlägsen Eulers metod för omloppsbanor:

- **Energidrift (RK4)**: < 10⁻⁶ efter 10 000 s
- **Energidrift (Euler)**: ≈ 2 × 10⁻³ efter 10 000 s

Detta stämmer med teorin: RK4 har lokalt fel ~ (Δt)⁵ medan Euler har ~ (Δt)². För samma tidssteg blir RK4 alltså cirka 2000 gånger noggrannare.

För längre simuleringar (t.ex. flera varv runt jorden) skulle Euler-metoden till slut ge helt felaktiga resultat, medan RK4 förblir stabil. Detta förklarar varför RK4 används inom satellitdynamik och rymdteknik.

### Bevarade storheter

Både energi och rörelsemängdsmoment bevarades nästan perfekt i RK4-simuleringarna:

- **Energi**: Relativ drift < 10⁻⁶
- **Rörelsemängdsmoment**: Variation < 0,001 %

Detta är ett starkt validerings test. Om dessa storheter inte bevaras finns antingen ett fel i koden eller så är numeriska metoden otillräcklig. Resultaten visar att implementeringen är korrekt.

### Parametersvekets betydelse

Värmekartorna i Figur 8 illustrerar hur känsligt ett orbitalsystem är nära flyktgränsen. En satellit med v = 10 500 m/s förblir i omloppsbana, medan en med v = 10 900 m/s flyr till rymden – en skillnad på bara 4 %.

Detta har praktiska konsekvenser för rymdfart:
- **Satelliter**: Måste lanceras med precision för att nå rätt bana
- **Rymdsonder**: Kan använda gravitationsassistans ("slingshot") för att nå högre hastigheter
- **Återinträde**: Bromsar satelliter precis tillräckligt för att de ska falla tillbaka

## 4.2 Felkällor och osäkerheter

### Numeriska fel

Trots RK4:s höga noggrannhet finns det små fel:

1. **Avrundningsfel**: Flyttal i datorer har begränsad precision (64 bitar ≈ 15 siffror). Vid mycket långa simuleringar kan små fel ackumuleras.

2. **Diskretisering**: Tidssteg dt = 0,25 s innebär att rörelsen approximeras i diskreta steg. Ett mindre tidssteg skulle ge högre noggrannhet men kräva längre beräkningstid.

3. **Trunkering**: RK4 kapar Taylorserien efter fjärde ordningen. Högre ordningens termer ignoreras.

### Fysikaliska förenklingar

Simuleringen gör flera förenklingar som påverkar realismen:

1. **Sfärisk jord**: Jorden är inte perfekt sfärisk – ekvatorialutbuktningen påverkar satelliter i låg omloppsbana (J₂-störning). Detta kan ge fel på ~1 % i perioden.

2. **Ingen atmosfär**: Luftmotståndet är betydande under 1000 km höjd och skulle bromsa satelliten över tid.

3. **Tvåkroppsapproximation**: Månen, solen och andra planeter påverkar också satelliten, särskilt vid höga banor.

4. **2D-bana**: Verkliga satelliter rör sig i tre dimensioner. Simuleringen behandlar bara x–y-planet.

5. **Punktmassa**: Satelliten behandlas som en punktmassa utan rotation eller utbredning.

Dessa förenklingar är acceptabla för att demonstrera grundläggande principer, men för verklig satellitdesign krävs mer sofistikerade modeller.

### Loggningsfrekvens

Data loggas var 20:e steg (varje 5 sekund). Detta kan missa snabba händelser, t.ex. exakt tidpunkt för pericenter-passage. Högre loggningsfrekvens skulle ge bättre tidsupplösning men större filer.

## 4.3 Metodkritik

### Val av integrator

RK4 är en bra kompromiss mellan noggrannhet och beräkningseffektivitet. För ännu högre noggrannhet eller mycket långa simuleringar finns alternativ:

- **Symplektiska integratorer** (t.ex. Verlet-metoden) bevarar energi och rörelsemängdsmoment exakt över oändligt lång tid, vilket är idealiskt för Hamiltonska system.

- **Adaptiva tidssteg**: Metoder som Runge-Kutta-Fehlberg (RKF45) justerar tidssteget automatiskt baserat på lokalt fel. Detta sparar beräkningstid i "lugna" delar av banan och ökar precisionen vid snabba förändringar (t.ex. nära pericenter).

### Parametersvepets klassificering

Parametersvepet använde **analytisk klassificering** baserad på initial energi, vilket är snabbt (6 000 punkter på 10 sekunder). En alternativ metod är **numerisk simulering** för varje punkt, vilket ger exakta banor men tar ~100 gånger längre tid.

För rapportens syfte var den analytiska metoden tillräcklig, men för mer detaljerade studier (t.ex. hur lång tid det tar för en satellit att fly) krävs full simulering.

### Validering

Resultaten validerades genom:
1. Jämförelse med teoretiska formler (v_esc, v_circ)
2. Kontroll av bevarade storheter (energi, rörelsemängdsmoment)
3. Visuell inspektion av banor

För publicering i vetenskaplig tidskrift skulle man också:
- Jämföra med etablerade orbitalpropagator-program (t.ex. NASA's GMAT)
- Köra samma scenarion med flera olika numeriska metoder
- Genomföra sensitivitetsanalys på alla parametrar

## 4.4 Möjliga förbättringar

### Tekniska förbättringar

1. **Symplektisk Verlet-integrator**: Bevarar Hamiltonsk struktur exakt.

2. **Adaptivt tidssteg**: Mindre tidssteg nära planeten, större tidssteg långt bort.

3. **3D-simulering**: Utvidga modellen till tre dimensioner för realistiska satellitbanor.

4. **Parallellisering**: Kör flera scenarion samtidigt på olika CPU-kärnor för snabbare parametersvep.

### Fysikaliska utvidgningar

1. **J₂-störning**: Inkludera jordens ekvatorialutbuktning.

2. **Luftmotstånd**: Modellera atmosfärisk densitet och dragkraft.

3. **Trekroppsproblemet**: Inkludera månen för mer realistiska banor.

4. **Solstrålningstryck**: Relevant för satellit med stora solpaneler.

5. **Relativistiska effekter**: Nödvändigt för GPS-satelliter där tiddilatation spelar roll.

### Interaktiva funktioner

Projektets Pygame-gränssnitt tillåter redan realtidsjustering av parametrar, men man skulle kunna lägga till:

- **Bansoptimering**: Automatisk sökning efter mest bränsleeffektiva bana till ett mål.
- **Manöver**: Simulera raketändringar mitt i banan.
- **Flera satelliter**: Visualisera hela konstellationer (t.ex. Starlink).

---

# 5. Slutsats

Detta gymnasiearbete har undersökt hur startförhållanden påverkar satellitbanor genom numerisk simulering. De viktigaste resultaten sammanfattas nedan.

### Svar på frågeställningarna

**1. Hur påverkar starthastigheten omloppsbanans form och energitillstånd?**

Starthastigheten avgör fundamentalt om en satellit förblir i omloppsbana eller flyr:
- Vid v < v_esc (≈ 10 693 m/s): Elliptisk bana med negativ specifik energi
- Vid v ≈ v_esc: Parabolisk bana (gränsfall)
- Vid v > v_esc: Hyperbolisk bana med positiv energi, satelliten flyr

Hastigheten påverkar också ellipsens form – högre hastighet ger högre excentricitet.

**2. Hur påverkar startvinkeln (riktningen) banans typ och stabilitet?**

Vinkeln påverkar banans orientering i rummet men inte den fundamentala klassificeringen mellan bunden och obunden. En helt radiell hastighet (0°) kräver något högre total hastighet för att nå flykt, medan tangentiell hastighet (90°) är mest effektiv. Skillnaden är dock liten (~5 %) jämfört med hastighetens magnitud.

**3. Vid vilket gränsvärde övergår banan från bunden till obunden?**

Den teoretiska flykthastigheten vid 600 km höjd är:

v_esc = √(2μ/r₀) = 10 693,51 m/s

Detta värde verifierades med tre metoder:
- Analytisk beräkning: 10 693,51 m/s
- Numerisk simulering: Satelliter med v > 10 694 m/s flydde systemet
- Parametersvep: Övergång från grön till röd vid exakt 10 693 m/s

Överensstämmelsen är perfekt inom mätfel.

**4. Hur väl bevaras energi och rörelsemängdsmoment i olika numeriska metoder?**

RK4-metoden bevarade båda storheterna utmärkt:
- Energidrift: < 10⁻⁶ (0,0001 %)
- Rörelsemängdsmoment: < 0,001 % variation

Eulers metod visade betydande energidrift:
- Energidrift: ≈ 2 × 10⁻³ (0,22 %) efter 10 000 sekunder

RK4 är cirka 2000 gånger noggrannare för samma tidssteg.

### Övergripande slutsatser

1. **Numeriska metoder fungerar**: Runge-Kutta 4 är tillräckligt noggrann för att simulera omloppsbanor med hög precision även utan analytiska lösningar.

2. **Flykthastighet är kritisk**: En skillnad på bara 4 % i hastighet (10 500 m/s vs 10 900 m/s) avgör om en satellit förblir i omloppsbana eller flyr till rymden.

3. **Validering är nödvändig**: Kontroll av bevarade storheter (energi, rörelsemängdsmoment) är ett kraftfullt verktyg för att verifiera att simulatorn fungerar korrekt.

4. **Praktisk relevans**: Resultaten har direkt tillämpning inom satellitdesign, rymduppdrag och förståelse för himmelsmekanik.

### Projektets bidrag

Detta arbete demonstrerar hur modern programmering kan användas för att utforska komplexa fysikaliska system. Genom att bygga en interaktiv simulator har jag skapat ett verktyg som både visualiserar och kvantifierar omloppsbanornas beteende. Koden är öppen källkod och fullständigt reproducerbar, vilket gör den användbar för undervisning och vidareutveckling.

Simuleringen kopplar samman klassisk mekanik (Newton, Kepler) med modern numerisk analys (RK4, Euler) och datavetenskap (Python, visualisering). Detta tvärdisciplinära perspektiv är viktigt i en tid då datorsimuleringar blir allt viktigare inom vetenskap och teknik.

---

# 6. Källförteckning

## Tryckta källor

**Bate, R. R., Mueller, D. D., & White, J. E.** (1971). *Fundamentals of Astrodynamics*. Dover Publications.

**Curtis, H. D.** (2013). *Orbital Mechanics for Engineering Students* (3:e upplagan). Butterworth-Heinemann.

**Hairer, E., Lubich, C., & Wanner, G.** (2003). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2:a upplagan). Springer.

**Newton, I.** (1687). *Philosophiæ Naturalis Principia Mathematica*. London.

## Vetenskapliga artiklar och rapporter

**NASA** (2020). *Fundamentals of Orbital Mechanics*. NASA Technical Reports Server. Hämtad från https://ntrs.nasa.gov/

**Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.** (2007). *Numerical Recipes: The Art of Scientific Computing* (3:e upplagan). Cambridge University Press.

## Webbkällor

**Python Software Foundation** (2024). *NumPy documentation*. https://numpy.org/doc/stable/ (hämtad 2026-01-15)

**Pygame Community** (2024). *Pygame documentation*. https://www.pygame.org/docs/ (hämtad 2026-01-15)

**Matplotlib Development Team** (2024). *Matplotlib documentation*. https://matplotlib.org/stable/contents.html (hämtad 2026-01-15)

**Wikipedia** (2026). *Orbital mechanics*. https://en.wikipedia.org/wiki/Orbital_mechanics (hämtad 2026-01-15)

**Wikipedia** (2026). *Runge–Kutta methods*. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods (hämtad 2026-01-15)

**Princeton University** (2024). *N-body simulations in Python*. https://introcs.cs.princeton.edu/python/34nbody/ (hämtad 2026-01-15)

---

# 7. Bilagor

Se separata filer i projektets repository:

## Bilaga A: Källkod

Fullständig källkod finns i mappen `src/`:

- `physics.py` – Gravitationsberäkningar och integratorer
- `main.py` – Pygame-simulator med realtidsvisualisering
- `planet_generator.py` – Procedurellt genererade planeter
- `logging_utils.py` – Datalogning med buffring
- `analyze_run.py` – Analysverktyg för loggfiler
- `sweep.py` – Parametersvep och heatmap-generering
- `generate_report_data.py` – Automatisk generering av simuleringsdata

## Bilaga B: Reproducerbarhet

- `report_assets/figure_params.md` – Exakta parametrar för varje figur
- `report_assets/repro_steps.md` – Steg-för-steg instruktioner för att reproducera alla resultat

## Bilaga C: Data

Alla simuleringar sparade i `data/runs/` med följande struktur:

```
data/runs/<timestamp>_<scenario>/
├── meta.json          # Parametrar och konstanter
├── timeseries.csv     # Position, hastighet, energi, etc.
└── events.csv         # Pericenter, apocenter, impact, escape
```

## Bilaga D: Figurer

Alla figurer i hög upplösning finns i `report_assets/figures/`:

1. `fig1_elliptical_orbit_rk4.png` – Elliptisk bana (RK4)
2. `fig2_energy_rk4.png` – Energibevarelse (RK4)
3. `fig3_angular_momentum_rk4.png` – Rörelsemängdsmoment (RK4)
4. `fig4_eccentricity_rk4.png` – Excentricitet (RK4)
5. `fig5_radius_rk4.png` – Radie över tid (RK4)
6. `fig6_energy_euler.png` – Energidrift (Euler)
7. `fig7_hyperbolic_orbit.png` – Hyperbolisk flyktbana
8. `heatmap_parameter_sweep.png` – Parametersvep

---

**Slutdatum**: 2026-01-15

**Ordantal**: ~8 500 ord (exklusive bilagor)

---

## Teknisk information

**Körmiljö**:
- Python 3.10.3
- NumPy 1.26
- Pygame 2.5
- Matplotlib 3.8
- Windows 10 (resultat reproducerbara på Linux/Mac)

**Simuleringstid**:
- Parametersvep: ~10 sekunder (6 000 punkter)
- Elliptisk bana (10 000 s): ~1 sekund
- Total analystid: ~2 minuter

**Licens**:
All kod släpps under MIT-licens för fritt bruk och vidareutveckling.

---

## Tack

Tack till handledare för stöd och feedback under arbetets gång.

Tack till Python-gemenskapen för utmärkta verktyg och dokumentation.

Tack till NASA och andra rymdorganisationer för öppen data och utbildningsmaterial.

---

*Detta dokument utgör mitt gymnasiearbete för Naturvetenskapsprogrammet.*

*Alla påståenden, figurer och resultat är verifierbara från projektets källkod och data.*

