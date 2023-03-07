#!/bin/bash
# run cmake and put into release mode
cmake -D DEAL_II_DIR=~/Software/deal.II/installed

make release
