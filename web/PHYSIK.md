# spinoza – Differentialgleichungen und Lösungsverfahren

## Der Zustandsvektor

Der Ball wird durch 9 Größen beschrieben:

    s(t) = ( x, y, z,  vₓ, vᵧ, vᵤ,  ωₓ, ωᵧ, ωᵤ )
             ─────────  ─────────────  ─────────────
             Position   Geschwindigkeit    Spin

Das ergibt ein System von 9 gekoppelten gewöhnlichen DGLs erster Ordnung.


## Die 9 Differentialgleichungen

### Gruppe 1: Kinematik (trivial)

    dx/dt = vₓ
    dy/dt = vᵧ
    dz/dt = vᵤ

Position ändert sich mit der Geschwindigkeit – klar.


### Gruppe 2: Dynamik (hier steckt die Physik)

    dvₓ/dt = aₓ(v, ω)
    dvᵧ/dt = aᵧ(v, ω)
    dvᵤ/dt = aᵤ(v, ω)

Die Beschleunigung a setzt sich aus drei Kräften zusammen:

#### a) Gravitation

    a_grav = (0, 0, −9.81)  m/s²

Einfach nach unten, konstant.

#### b) Luftwiderstand (Drag)

    F_D = −½ · C_D · ρ · A · |v| · v

    → a_drag = F_D / m = −½ · C_D · ρ · A · |v| · v / m

Der Drag ist proportional zum Quadrat der Geschwindigkeit und wirkt
immer entgegen der Flugrichtung. Der Faktor |v|·v (nicht v²) stellt
sicher, dass die Richtung stimmt.

Werte: C_D = 0.40, ρ = 1.2 kg/m³, A = π·r² = π·0.02² m², m = 2.7g

#### c) Magnus-Effekt (Spin → Kurvenbahn)

    F_M = C_L · ρ · A · r · (ω × v)

    → a_magnus = F_M / m

Das Kreuzprodukt ω × v ist der Schlüssel: Die Magnus-Kraft steht
senkrecht auf SOWOHL der Flugrichtung als auch der Spin-Achse.

Beispiele:
  - Topspin (ωₓ < 0): Ball dreht vorwärts → Kraft nach UNTEN
    Der Ball taucht schneller ab und springt flacher.
  - Backspin (ωₓ > 0): Ball dreht rückwärts → Kraft nach OBEN
    Der Ball schwebt länger und springt steiler.
  - Sidespin (ωᵤ ≠ 0): Kraft seitlich → Ball kurvt nach links/rechts.

Werte: C_L = 0.60, r = 0.02 m


### Zusammen: Gesamtbeschleunigung

    a(v, ω) = a_grav + a_drag(v) + a_magnus(v, ω)

Das System ist nichtlinear – eine analytische Lösung gibt es nicht,
daher numerische Integration. Die Nichtlinearität hat zwei Ursachen:
  1. Drag: |v|·v ist quadratisch in v (schon ohne Spin nichtlinear)
  2. Magnus: ω × v koppelt zwei Unbekannte multiplikativ


### Gruppe 3: Spin-Abbremsung

    dωₓ/dt = −(k_spin / I) · ωₓ
    dωᵧ/dt = −(k_spin / I) · ωᵧ
    dωᵤ/dt = −(k_spin / I) · ωᵤ

Der Spin wird durch Luftreibung langsam abgebremst (Stokes-Torque).
Das ist eine einfache exponentielle Dämpfung.

Werte: k_spin = 5·10⁻⁷ N·m·s, I = ⅔·m·r² (Hohlkugel)

Der Spin ändert sich im Flug kaum – die Abklingzeit liegt bei ~2.4s,
ein typischer Flug dauert nur ~0.2–0.4s.


## Lösungsverfahren: Runge-Kutta 4. Ordnung (RK4)

### Warum nicht einfach Euler?

Euler-Verfahren: s_{n+1} = s_n + dt · f(s_n)

Das wäre 1. Ordnung – der Fehler pro Schritt ist O(dt²).
Bei dt = 0.5ms und 600 Schritten akkumuliert sich das.

### RK4: 4 Steigungen pro Schritt

Idee: Statt nur am Anfang des Intervalls die Steigung zu nehmen,
berechnet RK4 vier Steigungen und mittelt sie gewichtet:

    k₁ = f(tₙ, sₙ)                          ← Steigung am Anfang
    k₂ = f(tₙ + dt/2, sₙ + dt/2 · k₁)      ← Steigung in der Mitte (mit k₁)
    k₃ = f(tₙ + dt/2, sₙ + dt/2 · k₂)      ← Steigung in der Mitte (mit k₂)
    k₄ = f(tₙ + dt,   sₙ + dt · k₃)        ← Steigung am Ende

    sₙ₊₁ = sₙ + (dt/6) · (k₁ + 2·k₂ + 2·k₃ + k₄)

Jedes kᵢ ist selbst ein 9-komponentiger Vektor (pos, vel, omega).

### Fehlerordnung

Lokaler Fehler:  O(dt⁵) pro Schritt
Globaler Fehler: O(dt⁴) über die gesamte Simulation

Bei dt = 0.5ms = 5·10⁻⁴s:
  dt⁴ = 6.25·10⁻¹⁴

Das ist absurd genau für diese Anwendung. Selbst dt = 5ms wäre
noch hinreichend, aber 0.5ms erlaubt präzise Aufprall-Erkennung.


### Konkret im Code (integrator.rs / physics.js)

    fn rk4_step(state, dt):
        // k1: Steigung am aktuellen Punkt
        k1_pos   = state.vel
        k1_vel   = acceleration(state)        ← die 3 Kräfte
        k1_omega = angular_deceleration(state) ← Spin-Dämpfung

        // k2: halber Schritt mit k1
        s2 = BallState(pos + dt/2·k1_pos, vel + dt/2·k1_vel, ω + dt/2·k1_omega)
        k2_pos   = s2.vel
        k2_vel   = acceleration(s2)
        k2_omega = angular_deceleration(s2)

        // k3: halber Schritt mit k2
        s3 = BallState(pos + dt/2·k2_pos, vel + dt/2·k2_vel, ω + dt/2·k2_omega)
        ...analog...

        // k4: voller Schritt mit k3
        s4 = BallState(pos + dt·k3_pos, vel + dt·k3_vel, ω + dt·k3_omega)
        ...analog...

        // Gewichtete Summe
        return BallState(
            pos   + dt/6 · (k1_pos   + 2·k2_pos   + 2·k3_pos   + k4_pos),
            vel   + dt/6 · (k1_vel   + 2·k2_vel   + 2·k3_vel   + k4_vel),
            omega + dt/6 · (k1_omega + 2·k2_omega + 2·k3_omega + k4_omega),
        )

Pro Zeitschritt werden also 4× alle Kräfte berechnet.
Für 600 Schritte: 2400 Kraftberechnungen – trivial für CPU.


## Was NICHT per DGL gelöst wird: Der Aufprall

Der Bounce ist kein kontinuierliches Problem, sondern ein
instantanes Impulsproblem (bounce.rs / physics.js):

### Erkennung

Wenn v_z < 0 (Ball sinkt) und die Flugbahn die Tischebene
z = 0.76m kreuzt, wird per linearer Interpolation der exakte
Aufprallzeitpunkt bestimmt und der RK4 dorthin sub-gesteppt.

### Impulsmodell (Gardin / Haake & Goodwill)

1. Normalkomponente:  v'_z = −e_n · v_z     (e_n = 0.93)
   → Ball prallt mit 93% der Normalgeschwindigkeit zurück.

2. Tangentialkomponente: Kontaktpunkt-Geschwindigkeit
   v_contact = v_tangential + ω × r_contact

   Dann Fallunterscheidung:
   a) |Reibungsimpuls| ≤ μ·|Normalimpuls| → HAFTEN (Rollkontakt)
      Der Ball "greift" die Tischoberfläche.
   b) Sonst → GLEITEN (kinetische Reibung, μ = 0.25)
      Der Ball rutscht über die Oberfläche.

3. Spin-Update aus dem Tangentialimpuls:
   Δω = (r_contact × J_tangential) / I

   Topspin wird durch den Aufprall VERSTÄRKT (~150 → ~213 rad/s),
   weil die Reibung den Ball noch stärker zum Rollen bringt.
