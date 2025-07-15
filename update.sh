#!/bin/bash

SOURCE_DIR=~/Lab/islp/exercise

rm -rf docs
mkdir docs

shopt -s nullglob
for notebook in "$SOURCE_DIR"/*.ipynb; do
    filename=$(basename "$notebook")
    echo "Mengonversi: $filename"
    jupyter nbconvert --to markdown "$notebook" --output-dir docs
done

cp index.md docs
cp about.md docs

# Fungsi untuk membersihkan spam output trainer dari markdown
clean_markdown_output() {
    local file=$1

    # Buat file sementara
    tempfile="${file}.cleaned"

    # Filter baris spam
    grep -vE "^\s*(Sanity Checking:|Training: \||Validation: \|)" "$file" \
        | sed '/^\s*$/N;/^\s*\n\s*$/D' > "$tempfile"

    # Ganti file asli dengan versi bersih
    mv "$tempfile" "$file"
}

# Loop semua file markdown hasil konversi
for mdfile in docs/*.md; do
    filename=$(basename "$mdfile")

    # Lewati index.md
    if [[ "$filename" == "index.md" ]]; then
        continue
    fi

    # Bersihkan spam output dari trainer
    clean_markdown_output "$mdfile"

    title="${filename%.md}"
    clean_title="${title//_/ }"

    # Ambil 3 baris pertama sebagai deskripsi
    snippet=$(head -n 3 "$mdfile" | sed ':a;N;$!ba;s/\n/ /g' | sed 's/^/📌 /')

    cat <<EOF >> docs/index.md

---

### 📄 [$clean_title]($filename)

$snippet

🔗 [Lihat Markdown](./$filename)

EOF
done

