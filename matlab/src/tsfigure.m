function [gcf1,gca1] = tsfigure( figsz )
%
%
%
gcf1 = figure();
gcf1.PaperUnits = 'inches';
gcf1.PaperPosition = [0, 0, figsz(1), figsz(2)];
paperposition = gcf1.PaperPosition;
gcf1.PaperSize = paperposition([3,4])+1;
gca1 = subplot(1,1,1); hold on; 
