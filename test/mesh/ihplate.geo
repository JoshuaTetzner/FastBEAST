lc = 0.1;
lc1 = 0.03;
Point(1) = {0, 0, 0, lc1};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(5) = {3, 4, 1, 2};
Plane Surface(6) = {5};

