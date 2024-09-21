    document.addEventListener("DOMContentLoaded", function () {
      var loginForm = document.getElementById("loginForm");
      if (loginForm) {
        loginForm.addEventListener("submit", async function (event) {
          event.preventDefault(); // Prevent the form from submitting normally

          // Create URLSearchParams to hold form data in application/x-www-form-urlencoded format
          var formData = new URLSearchParams();
          formData.append('username', document.getElementById('email').value.trim());
          formData.append('password', document.getElementById('password').value.trim());

          try {
            // Send the login data to the backend
            let response = await fetch("/token", {
              method: "POST",
              headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
              },
              body: formData.toString(),
            });

            if (response.ok) {
              // Handle successful login
              let data = await response.json();
              // Store the access token in localStorage
              localStorage.setItem("accessToken", data.access_token);
              // Redirect to the main chat page
              window.location.href = "index.html";
            } else {
              // Handle login failure
              const mainContent = document.querySelector(".main-content");
              mainContent.classList.add("shake-animation");
              mainContent.addEventListener(
                "animationend",
                function () {
                  mainContent.classList.remove("shake-animation");
                },
                { once: true }
              );
            }
          } catch (error) {
            console.error("Error logging in:", error);
          }
        });
      } else {
        console.error("The login form was not found.");
      }
    });