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

# Loop untuk daftar file
for mdfile in docs/*.md; do
    filename=$(basename "$mdfile")

    # Lewati index.md sendiri
    if [[ "$filename" == "index.md" ]]; then
        continue
    fi

    title="${filename%.md}"
    clean_title="${title//_/ }"

    # Ambil 3 baris pertama isi markdown sebagai deskripsi singkat
    snippet=$(head -n 3 "$mdfile" | sed ':a;N;$!ba;s/\n/ /g' | sed 's/^/📌 /')

    cat <<EOF >> docs/index.md

---

### 📄 [$clean_title]($filename)

$snippet

🔗 [Lihat Markdown](./$filename)

EOF
done

