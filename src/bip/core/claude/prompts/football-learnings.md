# Football Betting Learnings — Claude Role C Validator Prompt

> **Cómo se usa este archivo**: Claude (Role C) lee este documento entero como contexto cada vez que valida un pick. Cada commit a este archivo genera un git SHA nuevo que se stampea en el campo `claude_reasoning` de cada Pick — auditoría perfecta de "qué prompt vio Claude cuando tomó esta decisión".
>
> **Cómo crece**: cada nuevo aprendizaje se documenta en la sección **Detailed Entries** con frontmatter estructurado (ID estable, mercado, status). Las reglas accionables se consolidan en la **Quick Reference** y el **Decision Matrix** (mantenidos sincronizados manualmente).
>
> **Idioma**: español. Las decisiones del validator también deben venir en español si el contexto del pick lo está.

---

## 1. Quick Reference — Active Rules

> Tabla TLDR. Cada regla tiene un ID estable (R-NNN) referenciado desde Detailed Entries. Ordenadas por frecuencia de aplicación esperada.

| ID    | Regla                                                                                | Trigger                                                                                  | Acción del validator              | Origen   |
|-------|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|-----------------------------------|----------|
| R-001 | **Aggregate Stat Seduction**                                                         | Pick justificado SOLO con stat agregada de temporada sin contexto reciente               | FLAG `aggregate_stat_seduction`   | Error #1 |
| R-002 | **Recent form ≥ línea para Over/Under**                                              | Pick Over/Under cuando promedio combinado de últimos 5 contradice la línea               | FLAG `recent_form_contradicts`    | Error #1 |
| R-003 | **Empate sirve a ambos → no apostar gol temprano**                                   | Pick Over 1H / Gol 1H cuando empate sirve a los dos equipos                              | REJECT `cautious_first_half`      | Error #3 |
| R-004 | **Equipo en zona peligro + local + últimas 10 jornadas**                             | Pick contra equipo peleando descenso jugando de local en jornadas finales                | REJECT `desperation_home_factor`  | Error #3 |
| R-005 | **H2H > 3 temporadas = ruido**                                                       | Pick justificado con stats H2H de hace más de 3 temporadas                               | FLAG `h2h_too_old`                | Error #3 |
| R-006 | **Parlay: solo patas ≥ 4/5 confianza**                                               | Pata de parlay con confianza explícita < 4/5                                             | REJECT `weak_parlay_leg`          | Error #3 |
| R-007 | **EV positivo no valida análisis débil**                                             | Pick con EV positivo pero probabilidad inicial sin sustento contextual                   | FLAG `ev_circular_validation`     | Error #3 |
| R-008 | **2 bajas defensivas = ajuste cuantitativo**                                         | Bajas mencionadas como "riesgo" pero sin ajuste numérico a la prob. de gol del rival     | FLAG `injury_qualitative_only`    | Aprendizaje #2 |
| R-009 | **Equipo elite (top 3-4 de las grandes) puede romper patrones de visitante**         | Pick "visitante no marca" cuando el visitante es elite de Premier/LaLiga/Serie A/Bundes  | FLAG `elite_visitor_underrated`   | Aprendizaje #4 |
| R-010 | **Stats de liga ≠ stats europeas**                                                   | Pick que aplica stats de liga doméstica al contexto de copas europeas                    | FLAG `league_vs_europe_mismatch`  | Aprendizaje #4 |
| R-011 | **Equipos sin delanteros titulares: degradar ML → DC**                               | Pick ML para equipo que perdió a sus 1-2 delanteros titulares                            | FLAG `striker_absent_ml_risk`     | Aprendizaje #5 |
| R-012 | **Verificar definición exacta del mercado**                                          | Mercado nombrado con sigla ambigua (VA+2, AH+2, HC) sin verificación previa              | FLAG `market_definition_ambiguous`| Aprendizaje #6 |
| R-013 | **Motivación europea de local ≠ motivación liga**                                    | Pick contra equipo modesto jugando Europa en casa (especialmente raras participaciones)  | FLAG `european_home_motivation`   | Aprendizaje #4 |
| R-014 | **Confirmation bias: empezar por la cuota es señal de alarma**                       | Análisis menciona "cuota parece value" como justificación de búsqueda de datos           | FLAG `confirmation_bias`          | Error #3 |
| R-015 | **Simplicidad > complejidad cuando el contexto estructural es claro**                | Pick sobreanalizado granular cuando una tesis estructural simple ya da la respuesta      | CONFIRM (no penalty), nota         | Aprendizaje #4 |

**Cómo aplica el validator**: si el pick activa 1 trigger → FLAG con el `reason_code` correspondiente. Si activa 2+ triggers de severidad alta (R-003, R-004, R-006), → REJECT. Si no activa ninguno y el análisis está limpio → CONFIRM.

---

## 2. Decision Matrix por Mercado

> Mapa de "qué reglas aplicar primero según el tipo de mercado del pick".

### 2.1 Over/Under (totales de gol)
- **Aplicar siempre**: R-001, R-002, R-008
- **Aplicar si es 1H**: R-003 (empate sirve a ambos)
- **Aplicar si involucra Champions/Europa League**: R-010

### 2.2 1X2 / Match Winner
- **Aplicar siempre**: R-001, R-004, R-008, R-011
- **Aplicar si es eliminatoria EU**: R-009, R-013
- **Aplicar si involucra equipo modesto local en Europa**: R-013

### 2.3 BTTS (ambos marcan)
- **Aplicar siempre**: R-001, R-002, R-008
- **Aplicar si visitante elite**: R-009

### 2.4 Doble Oportunidad (DC)
- **Aplicar siempre**: R-004, R-011
- **Aplicar si DC 1H en eliminatoria EU**: nota positiva (Aprendizaje #4 confirma estructuralmente fuerte)

### 2.5 Parlay (cualquier composición)
- **Aplicar siempre**: R-006, R-007
- **Aplicar a cada pata individualmente**: todas las reglas del mercado correspondiente
- **Si alguna pata viola R-006**: REJECT el parlay completo

### 2.6 Mercados especiales (VA, hándicaps, asiáticos)
- **Aplicar siempre primero**: R-012 — si el mercado no está claramente definido en este archivo, FLAG y pedir clarificación antes de validar

---

## 3. Checklist de motivación (obligatorio antes de CONFIRM)

> Heurística derivada del Error #3. Claude debe poder responder mentalmente estas 6 preguntas antes de emitir CONFIRM en cualquier pick.

1. ¿Qué necesita el equipo A de **este partido específico**? (3 pts obligatorios / 1 punto basta / nada)
2. ¿Qué necesita el equipo B?
3. **¿El empate le sirve a alguno de los dos? ¿A ambos?**
4. ¿Perder tiene consecuencias distintas a empatar para cada uno?
5. Si el empate sirve a ambos → **descartar mercados de gol temprano** (R-003)
6. ¿Hay factor desesperación (descenso, eliminación)? ¿Es de local? → **no apostar en contra** (R-004)

Si Claude no puede responder al menos 4/6 con certeza basada en el contexto del pick → FLAG con `reason_code='insufficient_motivation_context'`.

---

## 4. Reglas de ajuste cuantitativo

> Cuando el contexto del pick menciona bajas, ajustar la probabilidad implícita ANTES de calcular EV.

### Bajas defensivas
| Cantidad | Tipo | Ajuste a goles/P del rival |
|----------|------|----------------------------|
| 1 baja titular | Cualquier posición defensiva | +0.1 a +0.2 |
| 2 bajas | Misma línea (ej: pareja de centrales) | +0.3 a +0.5 |
| 3+ bajas | Defensa colapsada | +0.5 a +0.8, considerar FLAG por incertidumbre |

### Bajas ofensivas
| Situación | Acción |
|-----------|--------|
| 1-2 delanteros titulares fuera | Degradar de ML a DC (R-011) |
| Mediocampista creativo titular fuera | Reducir prob. Over 0.5 a +0.05-0.10 |

---

## 5. Detailed Entries (chronological)

> Cada entrada documenta un caso real que generó una o más reglas. Frontmatter estructurado para queryabilidad. **Nunca borrar**: si una regla se deprecia, mover a sección 6 sin tocar la entrada original.

### Entry-001 — Aggregate Stat Seduction

```yaml
id: ENTRY-001
date: 2026-04-05
type: error
market: Over/Under
liga: Liga BetPlay (Colombia)
fixture: Cúcuta Deportivo vs América de Cali
generated_rules: [R-001, R-002]
status: active
```

**Contexto:** Recomendé Over 2.5 goles @ 2.25 basándome principalmente en que Cúcuta tenía 1.80 GA/P en la temporada.

**Qué hice mal:**
- Tomé una stat agregada impactante (Cúcuta defensa entre las peores) y construí la narrativa sobre ella.
- Ignoré que los últimos 5 partidos de Cúcuta fueron: 0-0, 1-0, 2-2, 2-2, 0-2 → promedio solo 2.2 goles/P.
- Los últimos 5 de América: 2-0, 2-1, 0-0, 1-0, 1-1 → promedio solo 1.6 goles/P.
- **Promedio combinado reciente: 1.90 goles/P, BAJO la línea 2.5**.
- También ignoré que el line movement iba hacia Under (Over se alargó de +110 a +125).

**Consecuencia:** Pick con EV real negativo (~-5.5%) presentado como EV positivo (~+8-17%).

**Lección:**
> Una stat agregada impresionante (como GA/P de temporada) puede ser engañosa sin contraste temporal. Equipos ajustan defensiva/ofensivamente durante temporada. **Siempre cruzar agregado con forma reciente real de los partidos.**

**Señal de alerta temprana:** Si al construir el caso del pick solo tengo UNA stat fuerte a favor y todo lo demás es neutral o en contra, probablemente estoy pescando edge donde no hay.

---

### Entry-002 — Bajas defensivas múltiples = ajuste cuantitativo

```yaml
id: ENTRY-002
date: 2026-04-05
type: aprendizaje
market: BTTS
liga: Liga BetPlay (Colombia)
fixture: Cúcuta Deportivo vs América de Cali
generated_rules: [R-008]
status: active
```

**Contexto:** Mismo partido. Tras corregir Error #1, aposté BTTS No @ 1.80 (EV +8%) y gané. El proceso funcionó: +1.00u neto.

**Qué observé:**
- América viajaba sin 2 centrales titulares (Nicolás Hernández rodilla + Jan Lucumí ligamentos).
- Identifiqué esto como riesgo en el contra-argumento y bajé el stake de BTTS No a 1.25u.
- Pero NO ajusté numéricamente mi estimación de goles de Cúcuta por esas bajas.
- Resultado: Cúcuta metió 2 goles (por encima de su avg reciente de 1.4/P), América 0.

**Lección:**
> Cuando un equipo tiene **2+ bajas en la misma línea** (especialmente defensa central), el ajuste no debe ser solo cualitativo ("lo anoto como riesgo"). Debe ser cuantitativo — ver sección 4.

---

### Entry-003 — Parlay 3/3 fallido (4 sub-errores)

```yaml
id: ENTRY-003
date: 2026-04-06
type: error
market: Parlay
generated_rules: [R-003, R-004, R-005, R-006, R-007, R-014]
status: active
```

**Contexto:** Parlay de 3 patas:
1. Gol en 1H Napoli vs Milan @ 1.40 — PERDIDA
2. Villarreal ML @ 2.25 (Girona vs Villarreal) — PERDIDA
3. Benfica gana 1H @ ~1.75 (Casa Pia vs Benfica) — PERDIDA

#### Sub-error A: Motivación mal leída (Napoli vs Milan)

Asumí que "2do vs 3ro, 1 punto de diferencia" = partido abierto. Construí todo sobre el stat de 13/13 partidos de Napoli con gol antes del HT en casa.

Realidad: Inter lideraba por 9 puntos. Scudetto decidido. Napoli y Milan peleaban entre sí por el 2do puesto, donde **un empate no perjudicaba a ninguno** → incentivo a NO arriesgar en 1H.

> Lección: Confundí "partido grande" con "partido abierto". Empate sirve a ambos → 1H cautelosa. **Origina R-003.**

#### Sub-error B: Confirmation bias + stats descontextualizadas (Girona vs Villarreal)

Me enamoré de la cuota (2.25) y busqué datos que confirmaran: H2H desde 2012, away win rate de 43%. Descarté que Girona peleaba descenso de local.

> Lección: H2H >2-3 temporadas = ruido. Equipos peleando descenso de local en últimas 10 jornadas son significativamente más peligrosos. **Origina R-004 y R-005.**

#### Sub-error C: Pata débil incluida en parlay (Benfica 1H)

Marqué Benfica 1H con 3.5/5 de confianza, señalé 8 empates en 27 partidos (30% draw rate), y AÚN ASÍ la incluí.

> Lección: La cadena se rompe por el eslabón más débil. **Origina R-006.**

#### Sub-error D: EV como validación circular

Calculé probabilidades basadas en análisis superficial → obtuve EV positivo → usé el EV positivo como justificación.

> Lección: EV positivo no valida nada si la probabilidad de entrada es incorrecta. EV es output, no validación. **Origina R-007.**

---

### Entry-004 — DC primera mitad eliminatorias EU

```yaml
id: ENTRY-004
date: 2026-04-09
type: aprendizaje
market: Doble Oportunidad (DC)
liga: Europa League / Conference League
fixtures: Bologna, Freiburg, Porto, Mainz, AEK (cuartos ida)
generated_rules: [R-009, R-010, R-013, R-015]
status: active
```

**Contexto:** Parlay de 5 patas DC primera mitad (local no pierde HT) en cuartos de final ida EL/CL.

**Resultados:** mayoría acertó (Freiburg ganaba HT, Porto 1-1 HT, Mainz ganaba HT).

**Qué hice mal:**
1. Sobreanalicé y generé parálisis. La tesis simple "idas de cuartos = primeros tiempos sin goleadas" era correcta y la complicaba.
2. Confundí mercados (HC europeo ≠ HC asiático ≠ VA+2).
3. Subestimé a Rayo Vallecano y sobreestimé a AEK. Rayo (13° La Liga) ganó 3-0 a AEK en Conference League.
4. Sobreestimé fortaleza defensiva de Bologna. Aplique stats Serie A a contexto europeo. Villa metió 2 goles.

**Lecciones:**
- **4A:** DC primera mitad en eliminatorias EU = mercado estructuralmente sólido. Confiar en la tesis contextual.
- **4B:** No confundir mercados — verificar SIEMPRE el tipo exacto. **Origina R-012 (Entry-006).**
- **4C:** Equipos elite (top 3-4 de las grandes) pueden romper patrones de visita en Europa. **Origina R-009.**
- **4D:** Motivación europea de local ≠ motivación de liga. **Origina R-013.**
- **4E:** Simplicidad > complejidad cuando el contexto es claro. **Origina R-015.**

---

### Entry-005 — Porto sin delanteros = empate, no derrota

```yaml
id: ENTRY-005
date: 2026-04-09
type: aprendizaje
market: ML / Doble Oportunidad
liga: Europa League
fixture: Porto vs Forest
generated_rules: [R-011]
status: active
```

**Contexto:** Porto vs Forest 1-1. Porto sin Samu Aghehowa y Luuk de Jong (delanteros titulares).

**Validación:** Identifiqué correctamente que Porto ML era riesgoso por las bajas ofensivas y recomendé Porto o X (DC) en vez de ML. El DC acertó.

> Regla confirmada: Cuando un equipo pierde a sus delanteros titulares, degradar de ML a DC. El equipo puede mantener solidez defensiva en casa pero carece de poder de gol para ganar. **Origina R-011.**

---

### Entry-006 — Verificación de mercados (VA+2 ≠ AH+2)

```yaml
id: ENTRY-006
date: 2026-04-09
type: aprendizaje
market: Mercados especiales
generated_rules: [R-012]
status: active
```

**Contexto:** Interpreté inicialmente VA+2 como Asian Handicap +2 (probabilidad ~95%). En realidad VA+2 (Victoria Anticipada) = "cobro anticipado si el equipo toma ventaja de 2 goles" = esencialmente ML con un bonus.

> Regla: Siempre verificar la definición exacta del mercado en la casa de apuestas. **La cuota lo delata**: si un AH +2 paga >2.00, probablemente NO es hándicap. **Origina R-012.**

---

## 6. Deprecated Rules

> Reglas que se demostraron equivocadas con más data. Mover acá (no borrar) cuando una entrada genere una regla que reemplace a otra existente.

*Vacío por ahora.*

---

## 7. Template para nuevas entradas

> Copy-paste este bloque al final de la sección 5 cuando tengas un nuevo aprendizaje. Reemplazar `XXX` con el siguiente número secuencial.

```markdown
### Entry-XXX — [Título corto del aprendizaje]

```yaml
id: ENTRY-XXX
date: YYYY-MM-DD
type: error | aprendizaje | regla_nueva
market: 1X2 | Over/Under | BTTS | DC | Parlay | Mercados especiales
liga: [opcional — solo si la lección es liga-específica]
fixture: [opcional — equipo A vs equipo B]
generated_rules: [R-NNN, R-NNN]   # si origina reglas nuevas, agregarlas a sección 1 también
status: active
```

**Contexto:** [qué pick, cuál fue la justificación]

**Qué pasó / Qué hice [bien|mal]:** [hechos verificables]

**Lección:**
> [Una frase clara que pueda convertirse en regla operativa]

**Origina/refuerza regla:** R-NNN (si es nueva, agregarla a sección 1 con su trigger y acción)
```

### Workflow para agregar una entrada nueva

```bash
# 1. Identificar el aprendizaje (post-mortem de un pick concreto)
# 2. Editar este archivo:
#    - Agregar Entry-XXX en sección 5 usando el template
#    - Si origina regla nueva: agregar R-NNN en sección 1 con trigger + acción + reason_code
#    - Si actualiza Decision Matrix (sección 2): editar tabla del mercado correspondiente
#    - Si depreca regla existente: mover a sección 6, NO borrar de Detailed Entries
# 3. Commit
git add src/bip/core/claude/prompts/football-learnings.md
git commit -m "learnings(03-04): Entry-XXX — [título] (genera R-NNN)"

# 4. La próxima ejecución del validator detecta el nuevo SHA y lo stampea en claude_reasoning
```

### Reglas para mantener la escalabilidad

1. **IDs son inmutables**. Una vez asignado R-007 o ENTRY-005, nunca se reusan ni se renumeran. Si una regla se deprecia, su ID no se libera.
2. **Detailed Entries son archive-only**. Nunca editar una entrada existente — si la lección original era incompleta, agregar una nueva entrada que la complemente y cross-referencia con `(ver también ENTRY-XXX)`.
3. **Quick Reference y Decision Matrix son derivados** de Detailed Entries. Mantenerlos sincronizados manualmente — discrepancia = bug del prompt.
4. **No agregar reglas sin entrada que las soporte**. Cada regla debe trazearse a un caso real documentado en sección 5.
5. **Token budget**: este archivo debe mantenerse < 8K tokens (~32K chars) para que la cache ephemeral de Anthropic siga rentable. Cuando se acerque al límite, considerar dividir en sub-archivos por mercado (cambio estructural mayor — coordinarlo con el loader).
