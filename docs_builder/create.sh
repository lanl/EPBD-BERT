#!/usr/bin/sh

# to generate the doc, run the following line from terminal
# cd doc_builder/ && bash create.sh && cd ..


# pip install sphinx
# pip install sphinx-book-theme

# cd doc_builder/ && sphinx-quickstart && cd ..
    # > Separate source and build directories (y/n) [n]: y
    # > Project name: pyDNA-EPBD
    # > Author name(s): AK
    # > Project release []: 
    # > Project language [en]:

export PYTHONPATH=$PYTHONPATH:$(pwd)/epbd_bert # running this will have impact only when we run it

# cd doc_builder/ && make clean && cd ..
make clean # remove everything under build
rm -rf source/modules

# a module directory must have __init__.py in the directory
sphinx-apidoc -o source/modules ../epbd_bert/


make html

rm -rf ../doc/*
touch ../doc/.nojekyll
mv build/html/* ../doc/
mv build/doctrees ../doc/
rm -rf build