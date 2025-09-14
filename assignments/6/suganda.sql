CREATE TABLE pembeli (
    id_pembeli SERIAL PRIMARY KEY,
    nm_pembeli VARCHAR(100) NOT NULL,
    alamat TEXT,
    sex VARCHAR(10) CHECK (sex IN ('PRIA', 'WANITA'))
);

CREATE TABLE pakaian (
    id_pakaian SERIAL PRIMARY KEY,
    nm_pakaian VARCHAR(100) NOT NULL,
    jenis VARCHAR(50),
    ukuran VARCHAR(10)
);

CREATE TABLE membeli (
    id_pembeli INT NOT NULL,
    id_pakaian INT NOT NULL,
    PRIMARY KEY (id_pembeli, id_pakaian),
    FOREIGN KEY (id_pembeli) REFERENCES pembeli (id_pembeli) ON DELETE CASCADE,
    FOREIGN KEY (id_pakaian) REFERENCES pakaian (id_pakaian) ON DELETE CASCADE
);

SELECT *
FROM pembeli
WHERE alamat = 'Palu';

SELECT *
FROM pakaian
WHERE jenis = 'T-Shirt';

SELECT 
    p.id_pembeli,
    p.nm_pembeli,
    p.alamat,
    pk.id_pakaian,
    pk.nm_pakaian,
    pk.jenis,
    pk.ukuran
FROM pembeli p
JOIN pembeli_pakaian pp ON p.id_pembeli = pp.id_pembeli
JOIN pakaian pk ON pp.id_pakaian = pk.id_pakaian;