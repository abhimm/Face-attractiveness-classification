function showface(face)
figure;
imagesc(reshape(face,86,86) * 255 * 1000);
colormap(gray);
