pip freeze > requirements.txt
while read -r package; do
    name=$(echo "$package" | cut -d= -f1)
    version=$(echo "$package" | cut -d= -f2-)
    poetry add "$name@$version"
done < requirements.txt
