#!/usr/bin/env bash
set -euo pipefail

# --- 1. Verificación de Credenciales ---
# El script se detendrá si la variable GITHUB_TOKEN no está definida.
: "${GITHUB_TOKEN:?La variable de entorno GITHUB_TOKEN no está definida. Asegúrate de haberla guardado en tu ~/.bashrc}"

# --- 2. Configuración de Credenciales Temporales ---
echo "Configurando credenciales temporales para Git..."
# Limpiar cualquier configuración de credenciales previa para evitar conflictos.
git config --global --unset-all credential.helper || true
# Crear un archivo temporal seguro para las credenciales.
CREDS_FILE=$(mktemp)
# Configurar Git para que use este archivo.
git config --global credential.helper "store --file ${CREDS_FILE}"
# Escribir el token en el archivo. Se usa "x-access-token" como nombre de usuario genérico para tokens.
printf "https://%s:%s@github.com\n" "x-access-token" "$GITHUB_TOKEN" > "$CREDS_FILE"

# --- 3. Limpieza Automática ---
# La función 'cleanup' se ejecutará al salir del script (ya sea por éxito, error o Ctrl+C).
cleanup() {
  echo "Limpiando credenciales temporales..."
  rm -f "${CREDS_FILE}"
  git config --global --unset-all credential.helper
}
trap cleanup EXIT

# --- Lógica de Release ---
echo "Iniciando el proceso de release..."
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: Tienes cambios locales sin guardar (uncommitted)."
  exit 1
fi

echo "Sincronizando tags con el repositorio remoto..."
git fetch origin --tags

LAST_TAG=$(git tag --list 'v*.*.*' | sort -V | tail -n 1 2>/dev/null || echo "v0.0.0")
BASE_NUM=${LAST_TAG#v}
IFS='.' read -r MAJOR MINOR PATCH <<<"$BASE_NUM"
NEW_PATCH=$((PATCH + 1))
NEW_TAG="v${MAJOR}.${MINOR}.${NEW_PATCH}"

echo ""
echo "Preparando nueva versión:"
echo "Última versión encontrada: $LAST_TAG"
echo "Nueva versión a crear:     $NEW_TAG"
echo ""
read -rp "¿Proceder a crear y subir la etiqueta? (y/n): " OK
[[ "$OK" == "y" ]] || { echo "Cancelado por el usuario."; exit 1; }

echo "Creando etiqueta $NEW_TAG..."
git tag -a "$NEW_TAG" -m "Release $NEW_TAG"
echo "Subiendo etiqueta $NEW_TAG a GitHub (usando credenciales temporales)..."
git push origin "$NEW_TAG"
echo "¡Release completado!"