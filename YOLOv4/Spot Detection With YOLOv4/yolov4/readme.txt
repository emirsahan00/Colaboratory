1.İndirmeyi istediğimiz dosyanın içine sağ tık open git bash here --> Git clone https://github.com/AlexeyAB/darknet
2.Makefile içindeki 1. 2. ve 4. eşitlikleri 1 yaptık
3.Transfer Learning için yolov4.conv.137 ve yolov4.weights dosyalarını indirdik(github adresinden)
4.yolov4.cfg uzantılı olan dosyada değişiklikler yaptık bu değişiklikler;

Batch : Her bir iterasyonda alacağımız resim sayısı 
Subdivisons : Her bir batch'i kaç alt adıma böleceğimiz anlamına gelir (Yükseltmek eğitimi uzatır düşürmekte kaliteli eğitim yapmamıza engel olur) 
Max_batch : Her iterasyonda alacağı max resim sayısı
Steps : Max_batch'in %80-%90 aralığında olmalı 
Classes : Kaç sınıf olacağı(insan,araba vb.) !bütün classesleri!
Filters : class'ın 5 fazlasının 3 katı olmalı !bütün filtersşları!