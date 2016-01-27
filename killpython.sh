#! /bin/bash

pgrep 'python' | xargs kill -9
pgrep 'say' | xargs kill -9