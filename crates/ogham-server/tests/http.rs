use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use ogham_server::app;
use serde_json::json;
use tower::ServiceExt;

#[tokio::test]
async fn health_ok() {
    let response = app()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("ok"));
}

#[tokio::test]
async fn compress_roundtrip() {
    let response = app()
        .oneshot(
            Request::post("/compress")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"messages": [{"role": "tool", "content": "[{\"a\":1}]"}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["messages"].as_array().unwrap().len(), 1);
    assert!(json["stats"]["original_tokens"].is_number());
    assert!(json["stats"]["compressed_tokens"].is_number());
    assert!(json["stats"]["ratio"].is_number());
    assert!(json["stats"]["compressor_used"].is_string());
}

#[tokio::test]
async fn retrieve_missing_is_found_false() {
    let response = app()
        .oneshot(
            Request::post("/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(json!({"id": "deadbeef"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["found"], false);
}

#[tokio::test]
async fn detect_json() {
    let response = app()
        .oneshot(
            Request::post("/detect")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"content": "[{\"a\":1},{\"a\":2}]"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["content_type"], "json_array");
}
