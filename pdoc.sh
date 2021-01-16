git checkout gh-pages
docker run --rm -v `pwd`:/home/joyvan/pygraphblas -it graphblas/pygraphblas-notebook:latest pdoc --html -f -o /home/joyvan/pygraphblas pygraphblas
git commit -m "Generate docs"
git push origin gh-pages
git checkout master
