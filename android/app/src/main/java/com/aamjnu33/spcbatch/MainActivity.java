package com.aamjnu33.spcbatch;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.ViewGroup;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

/** Full-screen WebView wrapper around the hosted Streamlit dashboard. */
public class MainActivity extends Activity {

    private WebView webView;
    private SwipeRefreshLayout swipe;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        webView = new WebView(this);
        webView.setBackgroundColor(Color.parseColor("#0E1117"));
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);      // Streamlit is a JS app
        s.setDomStorageEnabled(true);      // session state / local storage
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        s.setBuiltInZoomControls(true);    // pinch-zoom on wide charts
        s.setDisplayZoomControls(false);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri uri = req.getUrl();
                String host = uri.getHost() == null ? "" : uri.getHost();
                // Keep Streamlit (app + Community Cloud wake-up/auth pages) in-app;
                // open everything else (e.g. GitHub links) in the browser.
                if (host.endsWith("streamlit.app") || host.endsWith("streamlit.io")) {
                    return false;
                }
                startActivity(new Intent(Intent.ACTION_VIEW, uri));
                return true;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                swipe.setRefreshing(false);
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) {
                    swipe.setRefreshing(false);
                    view.loadDataWithBaseURL(null,
                            "<html><body style='background:#0E1117;color:#FAFAFA;font-family:sans-serif;"
                                    + "padding:32px'><h2>Can't reach the dashboard</h2>"
                                    + "<p>Check your connection, then pull down to retry.</p></body></html>",
                            "text/html", "utf-8", null);
                }
            }
        });

        swipe = new SwipeRefreshLayout(this);
        swipe.addView(webView, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        // Only pull-to-refresh when the page is scrolled to the top, so it
        // doesn't fight with scrolling inside the dashboard.
        swipe.setOnChildScrollUpCallback((parent, child) -> webView.getScrollY() > 0);
        swipe.setOnRefreshListener(() -> webView.loadUrl(getString(R.string.app_url)));
        setContentView(swipe);

        if (savedInstanceState != null) {
            webView.restoreState(savedInstanceState);
        } else {
            webView.loadUrl(getString(R.string.app_url));
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        webView.saveState(outState);
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
