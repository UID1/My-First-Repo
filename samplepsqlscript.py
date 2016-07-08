#!/usr/bin/python2.7
#
# Small script to show Postgresql and Psycopg together
#

import psycopg2

try:
    conn = psycopg2.connect("dbname='pythondb', user='pythonuser', host='localhost', password='python'")
except:
    print "Database connection error"

