## Ficha del Dataset: SMS Spam Collection (UCI)

## 1. Identificación y Procedencia
- **Nombre del Corpus:** SMS Spam Collection Data Set
- **Fuente Oficial:** UCI Machine Learning Repository
- **Licencia:** Licencia Abierta para uso educativo e investigación (CC BY 4.0 o equivalente del repositorio UCI).
- **Responsable / Colector:** Tiago A. Almeida y José María Gómez Hidalgo.

## 2. Diccionario de Datos
| Columna | Tipo de Dato | Significado / Descripción | Valores Válidos |
| :--- | :--- | :--- | :--- |
| `label` | Categórico (String) | Variable objetivo que indica si el mensaje es legítimo o correo no deseado. | `ham`, `spam` |
| `text` | Texto (String) | Contenido íntegro del mensaje SMS recibido. | Texto libre en inglés |

## 3. Ejemplos Anonimizados
1. **`ham`**: "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
2. **`spam`**: "Free entry in 2 a weekly comp to win FA Cup final tickets 21st May 2005..."
3. **`ham`**: "Ok lar... Joking wif u oni..."
4. **`spam`**: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
5. **`ham`**: "U dun say so early hor... U c already then say..."

## 4. Sesgos, Limitaciones y Población No Cubierta
- **Idioma:** Exclusivamente en inglés, con modismos y abreviaciones típicas de la mensajería SMS de principios de los 2000.
- **Población No Cubierta:** Mensajes modernos de plataformas de mensajería cifrada (WhatsApp, Telegram) o textos multilingües.
- **Sesgo Potencial:** Desbalance natural de clases (la gran mayoría de los mensajes cotidianos son de tipo `ham`).
