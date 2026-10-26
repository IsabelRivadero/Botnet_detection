BotFarm - Sistema de Detección y Visualización de Granjas de Bots

BotFarm es una herramienta modular para la detección, evaluación y visualización de granjas de bots e influencias coordinadas en redes sociales mediante análisis estructural, temporal y semántico, respaldado por algoritmos de detección de anomalías (Isolation Forest y Local Outlier Factor).

🏗️ Arquitectura del Proyecto

botfarm/
├── data/                       # Datasets de entrada (usuarios y tweets)
├── reports/                    # Archivos estáticos generados (PNG e HTML)
├── src/botfarm/
│   ├── api/                    # API REST pública (FastAPI / Uvicorn)
│   ├── visualization/          # Generación de gráficos (Quadrant, Radar, Vis.js)
│   ├── cli.py                  # Interfaz de línea de comandos unificada
│   ├── config.py               # Configuración global del detector
│   ├── detector.py             # QuadrantDetector (IF + LOF)
│   ├── embedders.py            # HashingEmbedder para procesamiento de texto
│   ├── io.py                   # Carga e ingesta de datos
│   └── pipeline.py             # Extracción de características por comunidad
├── tests/                      # Suite de pruebas automatizadas (pytest)
├── Dockerfile                  # Configuración de imagen del contenedor
├── docker-compose.yml          # Despliegue del servicio API REST
└── pyproject.toml              # Dependencias y configuración del paquete


🚀 Inicio Rápido con Docker Compose

El método recomendado para desplegar el servidor de producción es utilizando Docker Compose:

Construir e iniciar el servidor:

docker-compose up --build -d


Verificar que la API esté activa:

curl -X GET "http://127.0.0.1:8000/api/v1/health"


Generar reportes e imágenes de visualización:

curl -X POST "http://127.0.0.1:8000/api/v1/visualize"


Respuesta esperada:

{
  "status": "success",
  "reports": {
    "quadrants_image": "/reports/cuadrantes_if_lof.png",
    "radar_image": "/reports/radar_explicabilidad.png",
    "interactive_network": "/reports/red_botnets_islas_grid.html"
  }
}


Acceder a la documentación OpenAPI (Swagger):
Abre http://localhost:8000/docs en tu navegador.

Detener el servicio:

docker-compose down


💻 Uso desde la CLI (Línea de Comandos)

Si deseas ejecutar el pipeline localmente sin Docker:

Entrenar el detector (fit):

python -m botfarm.cli fit --data-dir data --as-of 2013-04-30 --out-model model.joblib


Evaluar y clasificar comunidades (score):

python -m botfarm.cli score --data-dir data --as-of 2013-09-30 --model model.joblib --out-csv flagged_communities.csv


Generar reportes visuales (visualize):

python -m botfarm.cli visualize --input-csv flagged_communities.csv --out-dir reports


Iniciar la API REST localmente (api):

python -m botfarm.cli api --host 127.0.0.1 --port 8000


🧪 Pruebas Automatizadas

Para ejecutar la suite de pruebas automatizadas y asegurar la invarianza temporal:

pytest


📊 Endpoints Principales de la API

Método

Endpoint

Descripción

GET

/api/v1/health

Estado de salud del servicio API

POST

/api/v1/fit

Entrenamiento del modelo con dataset histórico

POST

/api/v1/score

Clasificación de comunidades

POST

/api/v1/visualize

Generación automática de gráficos y mapa interactivo

GET

/reports/{file_name}

Servidor estático para descargar e inspeccionar reportes e imágenes

🏷️ Cierre y Tag de Versión en Git

Una vez verificado el correcto funcionamiento de todo el sistema, puedes realizar el commit final y etiquetar la versión estable:

git add .
git commit -m "feat: completar despliegue con Docker, servidor estático y documentación README"
git tag -a v1.0.0 -m "Versión 1.0.0 estable de BotFarm"
