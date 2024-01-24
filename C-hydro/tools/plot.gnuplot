#set view map
#set pm3d at b map
#set dgrid3d 10,10,2
#splot "out.txt" u 1:2:3
plot "out.txt" u 1:2:3 w image
pause mouse close
