# 🧠 AskYourData - Entrenamiento personalizado de SQLCoder

Este proyecto adapta el modelo [`defog/sqlcoder-7b-2`](https://huggingface.co/defog/sqlcoder-7b-2) para interpretar preguntas en lenguaje natural y generar consultas SQL personalizadas sobre la base de datos `WideWorldImporters`, utilizando el enfoque eficiente de fine-tuning con LoRA.

---

## 📁 Estructura del Proyecto

```
AskYourData/
├── Models/
│   └── sqlcoder-7b-2/              # Modelo original descargado desde Hugging Face
├── DataSet/
│   └── WideWorldImporters/
│       ├── schema_prompt.txt       # Descripción detallada del esquema de la base de datos
│       └── train.txt               # Dataset con prompts y consultas SQL alineadas
├── main.py                         # Script para probar el modelo con preguntas en lenguaje natural
├── requirements.txt                # Lista de dependencias necesarias
├── README.md                       # Este archivo de documentación
└── venv/                           # Entorno virtual (no se sube al repositorio)
```

---

## 🔧 Archivos importantes

### 📄 `main.py`

Este archivo ejecuta el modelo local para transformar preguntas en lenguaje natural a SQL:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model_name = "./Models/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

natural_language_query = "Dame una consulta que devuelva los usuarios que se registraron en los últimos 30 días"
prompt = f"-- Escribe una consulta SQL para lo siguiente:\n{natural_language_query}\nSELECT"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2, do_sample=False, pad_token_id=tokenizer.eos_token_id)

sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nConsulta generada:")
print(sql_query)
```

---

### 📁 `DataSet/WideWorldImporters/`

Contiene los archivos necesarios para el entrenamiento del modelo con LoRA.

- `schema_prompt.txt`: describe las tablas, columnas, tipos de datos y relaciones en la base de datos WideWorldImporters. Se utiliza como contexto inicial.
  
  Ejemplo:
  ```
  Tabla: Customers
  Columnas: CustomerID (int, clave primaria), CustomerName (string), PhoneNumber (string), ...
  Relación: Customers.CustomerID -> Orders.CustomerID
  ```

- `train.txt`: contiene pares de pregunta + SQL para entrenamiento supervisado. Cada entrada tiene este formato:

  ```
  -- Escribe una consulta SQL para lo siguiente:
  ¿Cuáles son los productos más vendidos en el último trimestre?
  SELECT ...
  ```

---

## 📦 Dependencias (`requirements.txt`)

```text
accelerate==0.28.0
filelock==3.13.1
huggingface-hub==0.22.2
numpy==1.24.3
packaging==23.2
tokenizers==0.15.1
torch==2.2.1
transformers==4.39.3
```

---

## ⚙️ Instalación del entorno

### 1. Crear entorno virtual

```bash
python -m venv venv
```

### 2. Activar el entorno virtual

- En **Windows**:

```bash
venv\Scripts\activate
```

- En **Linux/macOS**:

```bash
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso del modelo (inferencia)

Asegúrate de que el modelo esté descargado en la carpeta `./Models/sqlcoder-7b-2`. Luego ejecuta:

```bash
python main.py
```

---

## 🧠 ¿Usa CPU o GPU?

El script detecta automáticamente si hay GPU (`cuda`) y carga el modelo en ese dispositivo. Si no hay GPU disponible, usa `cpu`.

---

## 🔁 Entrenamiento con LoRA (próximamente)

Se realizará fine-tuning usando el dataset personalizado y el archivo `schema_prompt.txt` como contexto. LoRA permite entrenar solo capas específicas del modelo para ahorrar memoria y recursos.

Pasos esperados:
1. Cargar modelo base de forma congelada
2. Aplicar capas LoRA a los heads de atención
3. Entrenar usando `train.txt`
4. Evaluar y guardar el modelo ajustado

---

## ✅ Estado actual

- [x] Modelo descargado localmente
- [x] Dataset creado (schema + prompts)
- [x] Script de inferencia funcionando
- [ ] Script de entrenamiento con LoRA (en desarrollo)

---

¡Listo para convertir preguntas en lenguaje natural en consultas SQL inteligentes! 💡
