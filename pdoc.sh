BRANCH=$(git symbolic-ref --short HEAD)
git checkout gh-pages
./pdoc.sh
git commit -a -m "Generate docs"
git push origin gh-pages
git checkout $BRANCH
