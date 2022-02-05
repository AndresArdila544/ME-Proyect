function [zrng] = initSeeds()
    zrng = [];
    for i = 1:100
        num = randi([1,2147483646]);
        zrng = [zrng num];
    end
    zrng;
end
