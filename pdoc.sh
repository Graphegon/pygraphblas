git checkout gh-pages
./pdoc.sh
git commit -m "Generate docs"
git push origin gh-pages
git checkout master
