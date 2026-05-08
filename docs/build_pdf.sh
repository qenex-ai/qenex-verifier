#!/bin/bash
# Regenerate the SI PDF and the one-page summary card from markdown sources.
# Requires: pandoc, xelatex, fontconfig (DejaVu Serif/Sans/Sans Mono fonts).
set -e
cd "$(dirname "$0")"

# Build the body from current verifier-subset markdown sources, mirroring
# what was used at v2.0.0 release. Headers in the source markdown are
# shifted DOWN one level to accommodate the consolidated SI's section
# numbering (#  →  ##  ; ##  →  ### ; etc.) — but only outside fenced
# code blocks (so that bash comments inside ```bash ... ``` stay as
# bash comments, not LaTeX section headings).
{
  echo
  echo '\newpage'
  echo
  echo '# 1. Abstract (v2 draft)'
  echo
  awk '
    /^```/ { incode = !incode; print; next }
    !incode && /^####/ { print "#####" substr($0, 5); next }
    !incode && /^###/  { print "####" substr($0, 4); next }
    !incode && /^##/   { print "###" substr($0, 3); next }
    !incode && /^#/    { print "##" substr($0, 2); next }
    { print }
  ' ../packages/qenex_chem/src/docs/V2_ABSTRACT.md

  echo
  echo '\newpage'
  echo
  echo '# 2. Supplementary materials manifest'
  echo
  awk '
    /^```/ { incode = !incode; print; next }
    !incode && /^####/ { print "#####" substr($0, 5); next }
    !incode && /^###/  { print "####" substr($0, 4); next }
    !incode && /^##/   { print "###" substr($0, 3); next }
    !incode && /^#/    { print "##" substr($0, 2); next }
    { print }
  ' ../SUPPLEMENTARY.md

  echo
  echo '\newpage'
  echo
  echo '# 3. v1 → v2 numeric reconciliation'
  echo
  awk '
    /^```/ { incode = !incode; print; next }
    !incode && /^####/ { print "#####" substr($0, 5); next }
    !incode && /^###/  { print "####" substr($0, 4); next }
    !incode && /^##/   { print "###" substr($0, 3); next }
    !incode && /^#/    { print "##" substr($0, 2); next }
    { print }
  ' ../packages/qenex_chem/src/docs/V1_VS_V2_RECONCILIATION.md

  echo
  echo '\newpage'
  echo
  echo '# 4. Verifier subset changelog'
  echo
  awk '
    /^```/ { incode = !incode; print; next }
    !incode && /^####/ { print "#####" substr($0, 5); next }
    !incode && /^###/  { print "####" substr($0, 4); next }
    !incode && /^##/   { print "###" substr($0, 3); next }
    !incode && /^#/    { print "##" substr($0, 2); next }
    { print }
  ' ../CHANGELOG.md
} > _body.md

# 1. Full SI PDF
cat _title.md _body.md > _combined.md
pandoc _combined.md \
  -o QENEX_LAB_v2_Supplementary.pdf \
  --pdf-engine=xelatex \
  --highlight-style=tango \
  --toc

# 2. One-page summary card
pandoc _summary.md \
  -o QENEX_LAB_v2_Summary_Card.pdf \
  --pdf-engine=xelatex

rm -f _combined.md
echo
echo "Generated:"
ls -la QENEX_LAB_v2_Supplementary.pdf QENEX_LAB_v2_Summary_Card.pdf
