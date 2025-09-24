#!/usr/bin/env bash
# Modo seguro: el script se detendrá si un comando falla.
set -euo pipefail

echo "✅ Iniciando el proceso de release..."

# 1. Asegurarse de tener los últimos tags de GitHub
echo "🔄 Sincronizando tags con el repositorio remoto..."
git fetch origin --tags

# 2. Calcular la siguiente versión
# Busca el último tag en la rama lab. Si no hay, empieza en v0.1.0
LAST_TAG=$(git describe --tags --abbrev=0 origin/lab 2>/dev/null || echo "v0.0.0")
BASE_NUM=${LAST_TAG#v}
IFS='.' read -r MAJOR MINOR PATCH <<<"$BASE_NUM"
NEW_PATCH=$((PATCH + 1))
NEW_TAG="v${MAJOR}.${MINOR}.${NEW_PATCH}"

# 3. Mostrar un resumen y pedir confirmación
echo ""
echo "🚀 Preparando nueva versión:"
echo "   Última versión encontrada en 'lab': $LAST_TAG"
echo "   Nueva versión a crear:            $NEW_TAG"
echo ""
read -rp "❓ ¿Proceder a crear y subir la etiqueta? (y/n): " OK
[[ "$OK" == "y" ]] || { echo "❌ Cancelado por el usuario."; exit 1; }

# 4. Crear y subir la nueva etiqueta de Git
echo "🏷️  Creando etiqueta $NEW_TAG..."
git tag -a "$NEW_TAG" -m "Release $NEW_TAG"

echo "📤 Subiendo etiqueta $NEW_TAG a GitHub..."
git push origin "$NEW_TAG"

echo "🎉 ¡Release completado! La etiqueta $NEW_TAG ha sido subida."
echo "   Revisa la pestaña 'Actions' en tu repositorio de GitHub para ver el workflow en ejecución."