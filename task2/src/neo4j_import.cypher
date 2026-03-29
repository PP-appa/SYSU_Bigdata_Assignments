// 先创建约束，避免重复节点
CREATE CONSTRAINT movie_id_unique IF NOT EXISTS
FOR (m:Movie) REQUIRE m.movie_id IS UNIQUE;

CREATE CONSTRAINT person_id_unique IF NOT EXISTS
FOR (p:Person) REQUIRE p.person_id IS UNIQUE;

CREATE CONSTRAINT genre_id_unique IF NOT EXISTS
FOR (g:Genre) REQUIRE g.genre_id IS UNIQUE;

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:OverviewEntity) REQUIRE e.entity_id IS UNIQUE;

// 导入 Movie 节点
LOAD CSV WITH HEADERS FROM 'file:///movie_nodes.csv' AS row
MERGE (m:Movie {movie_id: toInteger(row.movie_id)})
SET m.title = row.title,
    m.original_title = row.original_title,
    m.overview = row.overview,
    m.release_date = row.release_date,
    m.vote_average = toFloat(row.vote_average),
    m.vote_count = toInteger(row.vote_count),
    m.popularity = toFloat(row.popularity);

// 导入 Person 节点
LOAD CSV WITH HEADERS FROM 'file:///person_nodes.csv' AS row
MERGE (p:Person {person_id: toInteger(row.person_id)})
SET p.name = row.name;

// 导入 Genre 节点
LOAD CSV WITH HEADERS FROM 'file:///genre_nodes.csv' AS row
MERGE (g:Genre {genre_id: toInteger(row.genre_id)})
SET g.name = row.name;

// 导入 OverviewEntity 节点
LOAD CSV WITH HEADERS FROM 'file:///overview_entity_nodes.csv' AS row
MERGE (e:OverviewEntity {entity_id: toInteger(row.entity_id)})
SET e.name = row.name,
    e.label = row.label;

// 导入 BELONGS_TO 关系
LOAD CSV WITH HEADERS FROM 'file:///movie_genre_edges.csv' AS row
MATCH (m:Movie {movie_id: toInteger(row.movie_id)})
MATCH (g:Genre {genre_id: toInteger(row.genre_id)})
MERGE (m)-[:BELONGS_TO]->(g);

// 导入 ACTED_IN 关系
LOAD CSV WITH HEADERS FROM 'file:///acted_in_edges.csv' AS row
MATCH (p:Person {person_id: toInteger(row.person_id)})
MATCH (m:Movie {movie_id: toInteger(row.movie_id)})
MERGE (p)-[r:ACTED_IN]->(m)
SET r.character = row.character,
    r.cast_order = toInteger(row.cast_order);

// 导入 DIRECTED 关系
LOAD CSV WITH HEADERS FROM 'file:///directed_edges.csv' AS row
MATCH (p:Person {person_id: toInteger(row.person_id)})
MATCH (m:Movie {movie_id: toInteger(row.movie_id)})
MERGE (p)-[:DIRECTED]->(m);

// 导入 MENTIONS 关系
LOAD CSV WITH HEADERS FROM 'file:///movie_overview_entity_edges.csv' AS row
MATCH (m:Movie {movie_id: toInteger(row.movie_id)})
MATCH (e:OverviewEntity {entity_id: toInteger(row.entity_id)})
MERGE (m)-[:MENTIONS]->(e);
