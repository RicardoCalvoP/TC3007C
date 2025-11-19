# Reinforcement Learning Study Guide

## Maximum Likelihood — Explicación Completa y Paso a Paso

El **método de Máxima Verosimilitud (Maximum Likelihood Estimation, MLE)** es una técnica para **estimar los parámetros desconocidos** de una distribución probabilística a partir de datos observados.

En este documento verás:

1. Qué es una distribución y sus parámetros.
2. Qué es la verosimilitud y por qué la maximizamos.
3. De dónde salen las ecuaciones de μ y σ².
4. Cómo se derivan esas ecuaciones.
5. Cómo implementarlo paso por paso (incluyendo la forma secuencial que estás usando).

---

# 1. La distribución Gaussiana y sus parámetros

La distribución Normal o Gaussiana está definida por dos parámetros:

- **μ**: media real de la distribución.
- **σ²**: varianza real de la distribución.

La función de densidad de probabilidad (PDF) es:

\[
f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left({-\frac{(x-\mu)^2}{2\sigma^2}}\right)
\]

### Qué significa cada cosa:

- \(x\): un valor observado.
- \(\mu\): centro de la distribución, donde se concentra la mayor densidad.
- \(\sigma^2\): cuánto se dispersan los valores respecto a la media.
- Numerador \(1\): constante.
- Denominador \(\sqrt{2\pi\sigma^2}\): normalización para asegurar área total = 1.
- Exponente: mide qué tan lejos está x de μ (distancia cuadrática).

---

# 2. Qué es la verosimilitud (likelihood)

Para un conjunto de datos:

\[
x_1, x_2, \ldots, x_N
\]

la probabilidad de observar **exactamente esos valores** según la distribución es:

\[
L(\mu,\sigma^2)=\prod\_{i=1}^N f(x_i|\mu,\sigma^2)
\]

A esto se le llama **verosimilitud**.

Buscamos los valores de μ y σ² que **maximicen L** porque:

> Si la distribución está bien parametrizada, los datos reales deben tener alta densidad en ella.

---

# 3. Log-Verosimilitud (log-likelihood)

Trabajar con productos es incómodo, así que usamos el logaritmo:

\[
\ln L = \sum\_{i=1}^N \ln f(x_i|\mu,\sigma^2)
\]

Sustituyendo el PDF:

\[
\ln L =
-\frac{N}{2}\ln(2\pi)
-\frac{N}{2}\ln(\sigma^2)
-\frac{1}{2\sigma^2}\sum\_{i=1}^N (x_i - \mu)^2
\]

Notar:

- Los dos primeros términos dependen solo de σ².
- El último depende de μ y σ².

---

# 4. Derivación de μ (media)

Buscamos:

\[
\frac{\partial}{\partial\mu}\ln L = 0
\]

Solo la parte \((x_i -\mu)^2\) depende de μ.

\[
\frac{\partial}{\partial\mu}
\left[
-\frac{1}{2\sigma^2}\sum (x_i - \mu)^2
\right]
=
-\frac{1}{2\sigma^2}\sum -2(x_i-\mu)
\]

Simplificando:

\[
\sum (x_i - \mu)=0
\]

Reorganizas:

\[
N\mu=\sum x_i
\]

Entonces:

\[
\hat{\mu}=\frac{1}{N}\sum x_i
\]

---

# 5. Derivación de σ² (varianza)

Buscamos:

\[
\frac{\partial}{\partial\sigma^2}\ln L = 0
\]

Partimos de:

\[
\ln L = -\frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum (x_i - \mu)^2
\]

Derivando:

\[
\frac{\partial}{\partial\sigma^2}\ln L
= -\frac{N}{2}\frac{1}{\sigma^2} - \frac{1}{2(\sigma^2)^2}\sum (x_i - \mu)^2
\]

Igualamos a cero:

\[
-\frac{N}{2\sigma^2} - \frac{1}{2(\sigma^2)^2}
\sum (x_i - \mu)^2
= 0
\]

Multiplicando por \(2(\sigma^2)^2\):

\[
-N\sigma^2 + \sum (x_i -\mu)^2 = 0
\]

Entonces:

\[
\hat{\sigma}^2=\frac{1}{N}\sum (x_i -\hat{\mu})^2
\]

---

# 6. Forma secuencial (online) — La que tú usas

Las ecuaciones del PDF también muestran una forma **secuencial**, útil cuando no quieres volver a recorrer todos los datos:

### Media online:

\[
\mu*{t+1} = \mu_t + \frac{1}{N}(x*{t+1}-\mu_t)
\]

### Varianza online:

\[
\sigma^{2}_{t+1} = \sigma^{2}\_t + \frac{1}{N}\left[(x_{t+1}-\mu)^2 - \sigma^{2}\_t\right]
\]

Estas ecuaciones permiten estimar ambos parámetros **mientras recorres los datos**, y si aplicas varias iteraciones (epochs), μ y σ² convergen hacia los valores reales.

Esta es exactamente la lógica que usaste en tu código.

---

# 7. Pasos en orden para aplicar Maximum Likelihood a una Gaussiana

### **Paso 1: Recolectar los datos**

\[
x_1,x_2,\ldots,x_N
\]

### **Paso 2: Escribir la PDF**

\[
f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left({-\frac{(x-\mu)^2}{2\sigma^2}}\right)
\]

### **Paso 3: Definir la verosimilitud**

\[
L(\mu,\sigma^2) = \prod f(x_i|\mu,\sigma^2)
\]

### **Paso 4: Convertir a log-verosimilitud**

\[
\ln L =
-\frac{N}{2}\ln(2\pi)
-\frac{N}{2}\ln(\sigma^2)
-\frac{1}{2\sigma^2}\sum (x_i - \mu)^2
\]

### **Paso 5: Derivar con respecto a μ**

\[
\hat{\mu}=\frac{1}{N}\sum x_i
\]

### **Paso 6: Derivar con respecto a σ²**

\[
\hat{\sigma}^2=\frac{1}{N}\sum (x_i - \hat{\mu})^2
\]

### **Paso 7: (Opcional) Usar las versiones secuenciales**

\[
\mu*{t+1} = \mu_t + \frac{1}{N}(x_t-\mu_t)
\]
\[
\sigma*{t+1}^2 = \sigma_t^2 + \frac{1}{N}\left[(x_t-\mu)^2 - \sigma_t^2\right]
\]

### **Paso 8: Con los parámetros finales, calcular el PDF**

\[
PDF(x)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left({-\frac{(x-\mu)^2}{2\sigma^2}}\right)
\]

Esto describe la densidad de probabilidad de cada valor según tus datos.

---
