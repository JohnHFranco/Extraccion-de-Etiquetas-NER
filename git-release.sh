#!/usr/bin/env bash
set -euo pipefail

echo "✅ Iniciando el proceso de release..."

# --- VERIFICACIÓN DE CAMBIOS SIN GUARDAR (MEJORA DE SEGURIDAD) ---
if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ Error: Tienes cambios locales sin guardar (uncommitted)."
  echo "   Por favor, haz 'git commit' antes de crear una nueva versión."
  exit 1
fi

# 1. Sincronizar tags con el repositorio remoto
echo "🔄 Sincronizando tags con el repositorio remoto..."
git fetch origin --tags

# --- LÓGICA DE VERSIÓN MEJORADA ---
# Busca el último tag con formato v*.*.* en TODO el repositorio,
# lo ordena por versión y elige el último para evitar conflictos.
LAST_TAG=$(git tag --list 'v*.*.*' | sort -V | tail -n 1 2>/dev/null || echo "v0.0.0")
# --- FIN DE LA LÓGICA MEJORADA ---

BASE_NUM=${LAST_TAG#v}
IFS='.' read -r MAJOR MINOR PATCH <<<"$BASE_NUM"
NEW_PATCH=$((PATCH + 1))
NEW_TAG="v${MAJOR}.${MINOR}.${NEW_PATCH}"

# 3. Mostrar un resumen y pedir confirmación
echo ""
echo "🚀 Preparando nueva versión:"
echo "   Última versión encontrada: $LAST_TAG"
echo "   Nueva versión a crear:     $NEW_TAG"
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