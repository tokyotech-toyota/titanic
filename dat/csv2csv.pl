#!/usr/bin/perl -w

# USAGE:
#   perl csv2csv.pl train.csv > train.new.csv

use strict;
use warnings;

# print header
printf("# %11s,  %8s,  %6s,  %60s,  %6s,  %5s,  %5s,  %5s,  %20s,  %10s,  %15s,  %8s\n",
       "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked");

while(my $line = <>)
{
  if($line =~ /^PassengerId/){next;}  # skip first line
  # capture
  $line =~ /([0-9\.]*),([0-9\.]*),([0-9\.]*),(\".*\"),([\w ]*),([0-9\.]*),([0-9\.]*),([0-9\.]*),([^,]*),([0-9\.]*),([\w ]*),([\w ]*)/;
  # limit too long name
  my $_4 = $4;
  if(length($_4)>60){
    $_4 = substr($4, 0, 56) . "...\"";
  }
  # print
  printf("  %11s,  %8s,  %6s,  %60s,  %6s,  %5s,  %5s,  %5s,  %20s,  %10s,  %15s,  %8s\n",
         $1, $2, $3, $_4, $5, $6, $7, $8, $9, $10, $11, $12);
}
