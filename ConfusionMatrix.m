function [CMat_rel,CMat_abs] = ConfusionMatrix(Cprd,Cact)

Cprd_uq = unique(Cprd);
Cact_uq = unique(Cact);

NumPrd = length(Cprd_uq);
NumAct = length(Cact_uq);
% assert(NumPrd == NumAct)

% allocate memory
CMat_abs = NaN(NumPrd,NumAct);
CMat_rel = NaN(NumPrd,NumAct);
for j = 1:NumAct
    lgAct = Cact == Cact_uq(j);
    SumAct = sum(lgAct);
    for i = 1:NumAct
        lgPrd = Cprd == Cact_uq(i);

        Num = sum( lgPrd(lgAct) == true );
        CMat_abs(i,j) = Num;
        CMat_rel(i,j) = Num/SumAct;
    end
end
end